from __future__ import annotations

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

st.set_page_config(page_title="Modern Cliché — Dataset Collector", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.block-container { max-width: 1500px; padding-top: 2.2rem; }
.small-note { color: #666; font-size: 0.86rem; }
</style>
""", unsafe_allow_html=True)

st.title("Dataset Collector")
st.caption("Collect a large visual corpus first. Morphology analysis stays in the main Modern Cliché tool.")

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = None

with st.sidebar:
    st.header("Search")
    use_ddgs = st.checkbox("DDGS — Bing + DuckDuckGo", value=True)
    use_wikimedia = st.checkbox("Wikimedia Commons", value=True)
    use_openverse = st.checkbox("Openverse", value=True)
    ddgs_backend = st.selectbox("DDGS backend", ["auto", "bing", "duckduckgo"], index=0, disabled=not use_ddgs)
    ddgs_region = st.text_input("DDGS region", "us-en", disabled=not use_ddgs)
    ddgs_license = st.selectbox("DDGS license filter", ["No filter", "any", "Public", "Share", "ShareCommercially", "Modify", "ModifyCommercially"], index=0, disabled=not use_ddgs)
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

left, right = st.columns([1.25, 1])
with left:
    preset = st.checkbox("Include historical gate starter pack", value=True)
    query_text = st.text_area("Queries — one per line", value="torii\nhongsalmun\npaifang\nceremonial gateway", height=180)
with right:
    st.write("Query expansion")
    add_front = st.checkbox("+ front view", value=True)
    add_arch = st.checkbox("+ architecture", value=False)
    add_historic = st.checkbox("+ historic", value=False)
    st.markdown("<div class='small-note'>Start broad. Final resolution and perceptual-duplicate checks happen after download.</div>", unsafe_allow_html=True)

base_queries = clean_queries(query_text)
if preset:
    base_queries = clean_queries("\n".join(base_queries + GATE_STARTER_QUERIES))
suffixes = [name for enabled, name in [(add_front, "front view"), (add_arch, "architecture"), (add_historic, "historic")] if enabled]
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
                source_calls.append(("DDGS", lambda q=query: search_ddgs(q, int(max_per_query), region=ddgs_region.strip() or "us-en", backend=ddgs_backend, license_filter=None if ddgs_license == "No filter" else ddgs_license, photo_only=photo_only)))
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
        status.write(f"Indexed **{len(unique):,} unique candidate URLs** from {len(all_items):,} raw results.")
        if failures:
            with st.expander(f"{len(failures)} source/query failures"):
                st.code("\n".join(failures[:200]))

candidates: list[Candidate] = st.session_state.candidates
if candidates:
    raw_count = len(candidates)
    filtered = [c for c in candidates if candidate_passes_dimensions(c, int(min_short), int(min_long), allow_unknown)]
    domains = sorted({c.source for c in filtered})
    selected_sources = st.multiselect("Keep sources", domains, default=domains)
    filtered = [c for c in filtered if c.source in selected_sources]

    exclude_domains = clean_queries(st.text_area("Exclude page/image domains — one per line", value="", height=80))
    if exclude_domains:
        lowered = [d.casefold() for d in exclude_domains]
        filtered = [c for c in filtered if not any(d in (c.image_url + " " + c.page_url).casefold() for d in lowered)]

    a, b, c = st.columns(3)
    a.metric("Indexed", f"{raw_count:,}")
    b.metric("After metadata filter", f"{len(filtered):,}")
    known = sum(1 for item in filtered if item.width and item.height)
    c.metric("Known dimensions", f"{known:,}")

    st.download_button("Download candidate index CSV", data=export_candidate_index(filtered), file_name="candidate_index.csv", mime="text/csv", use_container_width=True)
    preview_rows = [item.record() for item in filtered[:500]]
    preview_df = pd.DataFrame(preview_rows)
    if not preview_df.empty:
        show_cols = [col for col in ["thumbnail_url", "source", "query", "title", "width", "height", "license", "provider", "page_url", "image_url"] if col in preview_df.columns]
        st.dataframe(preview_df[show_cols], use_container_width=True, height=560, column_config={"thumbnail_url": st.column_config.ImageColumn("preview", width="small"), "page_url": st.column_config.LinkColumn("page"), "image_url": st.column_config.LinkColumn("image")})
        if len(filtered) > 500:
            st.caption(f"Preview shows first 500 of {len(filtered):,}. All filtered candidates are still available for download.")

    st.divider()
    st.subheader("Download originals")
    cap = st.number_input("Maximum originals to attempt (0 = all)", 0, 10000, min(1000, len(filtered)), 50)
    to_download = filtered if cap == 0 else filtered[: int(cap)]
    st.caption("The downloader does not bypass logins, paywalls, access controls, or HTTP denials. It validates the actual raster dimensions and removes exact + perceptually near duplicates.")

    if st.button(f"Download and build dataset ({len(to_download):,} candidates)", use_container_width=True):
        if not to_download:
            st.warning("No candidates pass the current filters.")
        else:
            p = st.progress(0.0)
            line = st.empty()
            def report(done: int, total: int, label: str) -> None:
                p.progress(done / max(total, 1)); line.write(f"{done:,} / {total:,} — {label[:120]}")
            dataset_dir, metadata = download_dataset(to_download, OUTPUT_DIR, min_short_edge=int(real_min_short), min_long_edge=int(real_min_long), max_file_mb=int(max_file_mb), near_duplicate_distance=int(near_dup), progress=report)
            zip_path = make_zip(dataset_dir)
            st.session_state.last_dataset = (str(dataset_dir), str(zip_path))
            ok = sum(1 for r in metadata if r["status"] == "downloaded")
            duplicate = sum(1 for r in metadata if r["status"].startswith("duplicate"))
            failed = sum(1 for r in metadata if r["status"] == "failed")
            st.success(f"Saved {ok:,} originals. Removed {duplicate:,} duplicates. {failed:,} downloads failed.")

if st.session_state.last_dataset:
    dataset_path, zip_path = map(Path, st.session_state.last_dataset)
    if dataset_path.exists():
        st.info(f"Local dataset: `{dataset_path}`")
    if zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        if size_mb <= 250:
            st.download_button(f"Download ZIP ({size_mb:.1f} MB)", data=zip_path.read_bytes(), file_name=zip_path.name, mime="application/zip", use_container_width=True)
        else:
            st.warning(f"ZIP is {size_mb:.1f} MB. It was created locally but is too large for a safe in-browser download button.")
