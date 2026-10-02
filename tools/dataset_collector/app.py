from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from collector import (
    GATE_STARTER_QUERIES,
    Candidate,
    candidate_passes_dimensions,
    clean_queries,
    dedupe_candidates,
    download_dataset,
    expand_queries,
    export_candidate_index,
    make_zip,
    search_ddgs,
    search_openverse,
    search_wikimedia,
)

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("MC_COLLECTOR_OUTPUT", HERE / "output"))
CATEGORIES = [
    "Unsorted",
    "Torii",
    "Hongsalmun",
    "Paifang / Pailou",
    "Temple / Shrine Gate",
    "Monumental / Triumphal",
    "Stone / Post-and-Lintel",
    "Other",
]

st.set_page_config(page_title="Modern Cliché — Dataset Collector", layout="wide")
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.block-container { max-width: 1600px; padding-top: 2rem; }
.small-note { color: #666; font-size: 0.86rem; }
.thumb-title { font-size: 0.82rem; line-height: 1.2; min-height: 2.1rem; }
</style>
""",
    unsafe_allow_html=True,
)


def item_key(item: Candidate) -> str:
    return hashlib.sha1(item.image_url.encode("utf-8", errors="ignore")).hexdigest()[:14]


def inferred_category(item: Candidate) -> str:
    text = f"{item.query} {item.title}".casefold()
    if "torii" in text or "鳥居" in text:
        return "Torii"
    if "hongsalmun" in text or "홍살문" in text:
        return "Hongsalmun"
    if any(term in text for term in ["paifang", "pailou", "牌坊", "牌楼"]):
        return "Paifang / Pailou"
    if any(term in text for term in ["triumphal", "monumental gateway", "triumph"]):
        return "Monumental / Triumphal"
    if any(term in text for term in ["stone gateway", "post and lintel", "post-and-lintel", "trilith"]):
        return "Stone / Post-and-Lintel"
    if any(term in text for term in ["temple", "shrine", "torana", "gopuram"]):
        return "Temple / Shrine Gate"
    return "Unsorted"


def review_for(item: Candidate) -> dict:
    state = st.session_state.review.get(item.image_url)
    if state is None:
        state = {"keep": False, "category": inferred_category(item)}
        st.session_state.review[item.image_url] = state
    return state


def write_annotated_metadata(dataset_dir: Path, metadata: list[dict]) -> None:
    for row in metadata:
        state = st.session_state.review.get(row.get("image_url", ""), {})
        row["review_keep"] = bool(state.get("keep", False))
        row["review_category"] = state.get("category", "Unsorted")
    dataset_dir.joinpath("metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if metadata:
        columns = sorted({key for row in metadata for key in row.keys()})
        with dataset_dir.joinpath("metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(metadata)


if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = None
if "review" not in st.session_state:
    st.session_state.review = {}

st.title("Dataset Collector")
st.caption("Search broadly, curate visually in the browser, classify, then download only the selected originals.")

with st.sidebar:
    st.header("Sources")
    use_ddgs = st.checkbox("DDGS — Bing + DuckDuckGo", value=True)
    use_wikimedia = st.checkbox("Wikimedia Commons", value=True)
    use_openverse = st.checkbox("Openverse", value=True)
    ddgs_backend = st.selectbox("DDGS backend", ["auto", "bing", "duckduckgo"], index=0, disabled=not use_ddgs)
    ddgs_region = st.text_input("DDGS region", "us-en", disabled=not use_ddgs)
    ddgs_license = st.selectbox(
        "DDGS license filter",
        ["No filter", "any", "Public", "Share", "ShareCommercially", "Modify", "ModifyCommercially"],
        index=0,
        disabled=not use_ddgs,
    )
    photo_only = st.checkbox("Prefer photographs", value=True, disabled=not use_ddgs)
    max_per_query = st.number_input("Results / source / query", 10, 500, 100, 10)

    st.header("Candidate filter")
    min_short = st.number_input("Minimum short edge", 0, 10000, 700, 100)
    min_long = st.number_input("Minimum long edge", 0, 10000, 1200, 100)
    allow_unknown = st.checkbox("Keep unknown dimensions", value=True)

    st.header("Download validation")
    real_min_short = st.number_input("Actual minimum short edge", 0, 10000, 800, 100)
    real_min_long = st.number_input("Actual minimum long edge", 0, 10000, 1400, 100)
    near_dup = st.slider("Near-duplicate tolerance", 0, 12, 3)
    max_file_mb = st.number_input("Maximum file size (MB)", 1, 100, 25, 1)

search_tab, review_tab, export_tab = st.tabs(["01 SEARCH", "02 REVIEW + CLASSIFY", "03 BUILD DATASET"])

with search_tab:
    left, right = st.columns([1.25, 1])
    with left:
        preset = st.checkbox("Include historical gate starter pack", value=True)
        query_text = st.text_area(
            "Queries — one per line",
            value="torii\nhongsalmun\npaifang\nceremonial gateway",
            height=180,
        )
    with right:
        st.write("Query expansion")
        add_front = st.checkbox("+ front view", value=True)
        add_arch = st.checkbox("+ architecture", value=False)
        add_historic = st.checkbox("+ historic", value=False)
        st.markdown(
            "<div class='small-note'>Search can be intentionally broad. Visual curation happens in the next tab.</div>",
            unsafe_allow_html=True,
        )

    base_queries = clean_queries(query_text)
    if preset:
        base_queries = clean_queries("\n".join(base_queries + GATE_STARTER_QUERIES))
    suffixes = [
        name
        for enabled, name in [
            (add_front, "front view"),
            (add_arch, "architecture"),
            (add_historic, "historic"),
        ]
        if enabled
    ]
    queries = expand_queries(base_queries, suffixes)

    st.write(f"**{len(queries)} search queries** × selected sources × up to **{max_per_query} results** each")

    if st.button("Search candidates", type="primary", use_container_width=True):
        if not queries:
            st.error("Add at least one query.")
        elif not any([use_ddgs, use_wikimedia, use_openverse]):
            st.error("Select at least one source.")
        else:
            all_items: list[Candidate] = []
            tasks = len(queries) * sum([use_ddgs, use_wikimedia, use_openverse])
            completed = 0
            progress = st.progress(0.0)
            status = st.empty()
            failures: list[str] = []
            for query in queries:
                source_calls = []
                if use_ddgs:
                    source_calls.append(
                        (
                            "DDGS",
                            lambda q=query: search_ddgs(
                                q,
                                int(max_per_query),
                                region=ddgs_region.strip() or "us-en",
                                backend=ddgs_backend,
                                license_filter=None if ddgs_license == "No filter" else ddgs_license,
                                photo_only=photo_only,
                            ),
                        )
                    )
                if use_wikimedia:
                    source_calls.append(("Wikimedia", lambda q=query: search_wikimedia(q, int(max_per_query))))
                if use_openverse:
                    source_calls.append(("Openverse", lambda q=query: search_openverse(q, int(max_per_query))))
                for label, fn in source_calls:
                    status.write(f"Searching **{label}** — `{query}`")
                    try:
                        all_items.extend(fn())
                    except Exception as exc:
                        failures.append(f"{label} / {query}: {exc}")
                    completed += 1
                    progress.progress(completed / max(tasks, 1))
            unique = dedupe_candidates(all_items)
            st.session_state.candidates = unique
            st.session_state.review = {}
            status.write(f"Indexed **{len(unique):,} unique candidate URLs** from {len(all_items):,} raw results.")
            if failures:
                with st.expander(f"{len(failures)} source/query failures"):
                    st.code("\n".join(failures[:200]))

    if st.session_state.candidates:
        st.success(f"{len(st.session_state.candidates):,} unique candidates are ready for review.")

candidates: list[Candidate] = st.session_state.candidates
filtered: list[Candidate] = []
if candidates:
    filtered = [
        item
        for item in candidates
        if candidate_passes_dimensions(item, int(min_short), int(min_long), allow_unknown)
    ]

with review_tab:
    if not candidates:
        st.info("Search for candidates first.")
    else:
        source_options = sorted({item.source for item in filtered})
        controls = st.columns([1.2, 1.2, 1, 1])
        selected_sources = controls[0].multiselect("Sources", source_options, default=source_options)
        text_filter = controls[1].text_input("Find in title/query", "")
        page_size = controls[2].selectbox("Images / page", [24, 36, 48, 60], index=1)
        only_unreviewed = controls[3].checkbox("Only unreviewed", value=False)

        review_items = [item for item in filtered if item.source in selected_sources]
        if text_filter.strip():
            needle = text_filter.casefold().strip()
            review_items = [
                item
                for item in review_items
                if needle in f"{item.title} {item.query} {item.provider}".casefold()
            ]
        if only_unreviewed:
            review_items = [item for item in review_items if item.image_url not in st.session_state.review]

        reviewed_count = sum(1 for item in filtered if item.image_url in st.session_state.review)
        kept_count = sum(1 for state in st.session_state.review.values() if state.get("keep"))
        a, b, c, d = st.columns(4)
        a.metric("Indexed", f"{len(candidates):,}")
        b.metric("After metadata filter", f"{len(filtered):,}")
        c.metric("Reviewed", f"{reviewed_count:,}")
        d.metric("Kept", f"{kept_count:,}")

        total_pages = max(1, math.ceil(len(review_items) / page_size))
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (int(page) - 1) * page_size
        page_items = review_items[start : start + page_size]

        bulk_left, bulk_right, bulk_clear = st.columns([1, 1, 1])
        if bulk_left.button("Keep all on page", use_container_width=True):
            for item in page_items:
                key = item_key(item)
                state = review_for(item)
                state["keep"] = True
                st.session_state[f"keep_{key}"] = True
            st.rerun()
        if bulk_right.button("Reject all on page", use_container_width=True):
            for item in page_items:
                key = item_key(item)
                state = review_for(item)
                state["keep"] = False
                st.session_state[f"keep_{key}"] = False
            st.rerun()
        if bulk_clear.button("Reset page review", use_container_width=True):
            for item in page_items:
                key = item_key(item)
                st.session_state.review.pop(item.image_url, None)
                st.session_state.pop(f"keep_{key}", None)
                st.session_state.pop(f"cat_{key}", None)
            st.rerun()

        st.caption(f"Showing {start + 1:,}–{min(start + page_size, len(review_items)):,} of {len(review_items):,} filtered candidates.")

        columns = st.columns(4)
        for index, item in enumerate(page_items):
            col = columns[index % 4]
            key = item_key(item)
            state = review_for(item)
            with col:
                preview = item.thumbnail_url or item.image_url
                try:
                    st.image(preview, use_container_width=True)
                except Exception:
                    st.caption("Preview unavailable")
                title = (item.title or "Untitled").strip()
                if len(title) > 90:
                    title = title[:87] + "..."
                st.markdown(f"<div class='thumb-title'>{title}</div>", unsafe_allow_html=True)
                keep = st.checkbox("Keep", value=bool(state.get("keep", False)), key=f"keep_{key}")
                current_category = state.get("category", inferred_category(item))
                category = st.selectbox(
                    "Class",
                    CATEGORIES,
                    index=CATEGORIES.index(current_category) if current_category in CATEGORIES else 0,
                    key=f"cat_{key}",
                    label_visibility="collapsed",
                )
                state["keep"] = keep
                state["category"] = category
                st.session_state.review[item.image_url] = state
                dims = f"{item.width or '?'} × {item.height or '?'}"
                st.caption(f"{item.source} · {dims} · {item.query}")
                if item.page_url:
                    st.link_button("Source page", item.page_url, use_container_width=True)

        st.divider()
        review_rows = []
        for item in filtered:
            state = st.session_state.review.get(item.image_url)
            if not state:
                continue
            row = item.record()
            row.update({"review_keep": state.get("keep", False), "review_category": state.get("category", "Unsorted")})
            review_rows.append(row)
        if review_rows:
            review_df = pd.DataFrame(review_rows)
            st.download_button(
                "Download review CSV",
                data=review_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="candidate_review.csv",
                mime="text/csv",
                use_container_width=True,
            )

with export_tab:
    if not candidates:
        st.info("Search and review candidates first.")
    else:
        selected = [item for item in candidates if st.session_state.review.get(item.image_url, {}).get("keep")]
        st.metric("Selected originals", f"{len(selected):,}")

        category_counts: dict[str, int] = {}
        for item in selected:
            category = st.session_state.review.get(item.image_url, {}).get("category", "Unsorted")
            category_counts[category] = category_counts.get(category, 0) + 1
        if category_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"category": key, "count": value} for key, value in sorted(category_counts.items())]
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.download_button(
            "Download full candidate index CSV",
            data=export_candidate_index(candidates),
            file_name="candidate_index.csv",
            mime="text/csv",
            use_container_width=True,
        )

        cap = st.number_input(
            "Maximum selected originals to attempt (0 = all)",
            0,
            10000,
            min(1000, len(selected)) if selected else 0,
            50,
        )
        to_download = selected if cap == 0 else selected[: int(cap)]
        st.caption(
            "Only reviewed + kept images are downloaded. The downloader validates the real raster size and removes exact and perceptual near-duplicates."
        )

        if st.button(f"Build curated dataset ({len(to_download):,} selected)", type="primary", use_container_width=True):
            if not to_download:
                st.warning("Keep at least one candidate in the Review tab.")
            else:
                p = st.progress(0.0)
                line = st.empty()

                def report(done: int, total: int, label: str) -> None:
                    p.progress(done / max(total, 1))
                    line.write(f"{done:,} / {total:,} — {label[:120]}")

                dataset_dir, metadata = download_dataset(
                    to_download,
                    OUTPUT_DIR,
                    min_short_edge=int(real_min_short),
                    min_long_edge=int(real_min_long),
                    max_file_mb=int(max_file_mb),
                    near_duplicate_distance=int(near_dup),
                    progress=report,
                )
                write_annotated_metadata(dataset_dir, metadata)
                zip_path = make_zip(dataset_dir)
                st.session_state.last_dataset = (str(dataset_dir), str(zip_path))
                ok = sum(1 for row in metadata if row["status"] == "downloaded")
                duplicate = sum(1 for row in metadata if row["status"].startswith("duplicate"))
                failed = sum(1 for row in metadata if row["status"] == "failed")
                st.success(f"Saved {ok:,} originals. Removed {duplicate:,} duplicates. {failed:,} downloads failed.")

        if st.session_state.last_dataset:
            dataset_path, zip_path = map(Path, st.session_state.last_dataset)
            if dataset_path.exists():
                st.info(f"Dataset: `{dataset_path}`")
            if zip_path.exists():
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                if size_mb <= 250:
                    st.download_button(
                        f"Download ZIP ({size_mb:.1f} MB)",
                        data=zip_path.read_bytes(),
                        file_name=zip_path.name,
                        mime="application/zip",
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        f"ZIP is {size_mb:.1f} MB. It was created on the server, but is too large for a safe in-browser download button."
                    )
