from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modern_cliche.data import search_images, download_images, decode_image
from modern_cliche.segmentation import overlay
from modern_cliche.pipeline import prepare, subset, fingerprint, load_experiment
from modern_cliche.analysis import analyse, ml_available, visual_embeddings
from modern_cliche.depth import infer_depth, moge_available
from modern_cliche.morphology import aggregate_depth_results, coverage_preview, variation_preview
from modern_cliche.surface import surface_mesh
from modern_cliche.export import experiment_zip, sample_table, mesh_bytes, dumps

APP_TITLE = "MODERN CLICHÉ"
STEPS = ["COLLECT", "CURATE", "DISCOVER", "RECONSTRUCT", "EXPORT"]
FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif'

st.set_page_config(page_title="Modern Cliché", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    f"""
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    button, input, textarea, select, .stMarkdown, .stText, .stCaption {{ font-family: {FONT} !important; }}
    .block-container {{ max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }}
    h1 {{ font-size: 2.1rem !important; letter-spacing: -.04em; margin-bottom: .15rem !important; }}
    h2 {{ font-size: 1.45rem !important; letter-spacing: -.025em; margin-top: .5rem !important; }}
    h3 {{ font-size: 1.05rem !important; }}
    [data-testid="stMetricValue"] {{ font-family: {FONT} !important; }}
    .mc-step {{ font-size:.72rem; letter-spacing:.12em; color:#8a8a8a; padding:.35rem 0 .55rem; border-bottom:1px solid #e8e8e8; }}
    .mc-step.active {{ color:#111; border-bottom:2px solid #111; font-weight:700; }}
    .mc-note {{ color:#666; font-size:.88rem; line-height:1.45; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state():
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("seed", 17)
    st.session_state.setdefault("depth_cache", {})


def _clear_downstream(from_step: int):
    keys = {
        1: ["prepared", "inclusion", "curation_indices", "analysis", "used", "selected_group", "aggregate", "mesh", "geometry", "recipe", "ready_zip"],
        2: ["analysis", "used", "selected_group", "aggregate", "mesh", "geometry", "recipe", "ready_zip"],
        3: ["selected_group", "aggregate", "mesh", "geometry", "recipe", "ready_zip"],
        4: ["aggregate", "mesh", "geometry", "recipe", "ready_zip"],
    }
    for key in keys.get(from_step, []):
        st.session_state.pop(key, None)


def replace_dataset(samples, query, failures=None, overrides=None, manifest=None):
    for key in list(st.session_state):
        if key.startswith(("include_", "mask_edit_")):
            del st.session_state[key]
    for key in ["prepared", "inclusion", "curation_indices", "analysis", "used", "selected_group",
                "aggregate", "mesh", "geometry", "recipe", "ready_zip"]:
        st.session_state.pop(key, None)
    st.session_state.update(
        samples=samples, query=query, download_failures=failures or [],
        overrides=overrides or {}, restored=manifest or {}, step=1,
    )


def go(step: int):
    st.session_state.step = max(0, min(step, len(STEPS) - 1))
    st.rerun()


def step_header():
    columns = st.columns(len(STEPS), gap="small")
    for i, (column, name) in enumerate(zip(columns, STEPS)):
        cls = "mc-step active" if i == st.session_state.step else "mc-step"
        column.markdown(f'<div class="{cls}">{i + 1:02d} &nbsp; {name}</div>', unsafe_allow_html=True)


def quality_score(info: dict) -> float:
    foreground = float(info.get("foreground_fraction", .5))
    border = float(info.get("border_contact", 0.))
    discarded = float(info.get("discarded_fraction", 0.))
    score = 1.0
    score -= min(.45, border * 2.8)
    score -= min(.25, discarded * .8)
    if foreground < .015 or foreground > .88:
        score -= .45
    elif foreground < .035 or foreground > .80:
        score -= .15
    if info.get("needs_review"):
        score -= .15
    return float(np.clip(score, 0, 1))


def default_include(info: dict) -> bool:
    return quality_score(info) >= .58


def figure_mesh(mesh):
    vertices, faces = mesh.vertices, mesh.faces
    figure = go.Figure(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#bdbdbd", flatshading=False,
        lighting=dict(ambient=.45, diffuse=.75, roughness=.65, specular=.12),
    ))
    figure.update_layout(
        height=600, margin=dict(l=0, r=0, t=0, b=0), font=dict(family=FONT),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X", showbackground=False),
            yaxis=dict(title="Depth", showbackground=False),
            zaxis=dict(title="Y", showbackground=False),
            camera=dict(eye=dict(x=1.1, y=-1.8, z=.8)),
        ),
    )
    return figure


@st.cache_data(ttl=1800, max_entries=12, show_spinner=False)
def cached_search(query, limit, provider):
    return search_images(query, limit, provider)


_init_state()
st.title(APP_TITLE)
st.caption("DATA-DRIVEN MORPHOLOGY · IMAGE → CLUSTER → VISIBLE 3D")
step_header()
st.write("")

if st.session_state.step == 0:
    st.header("Collect a broad image cohort")
    st.markdown(
        '<div class="mc-note">Start wide. The app will remove duplicates, reject weak masks and discover visual groups later. '
        'Search input may mix Korean and English; known Korean object words are translated while the rest of the query is preserved.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    search_tab, upload_tab, restore_tab = st.tabs(["SEARCH", "UPLOAD", "RESTORE"])
    with search_tab:
        with st.form("search_form"):
            query = st.text_input("Subject", "gargoyle", placeholder="gargoyle / 가고일 church / cat sculpture")
            a, b = st.columns(2)
            provider = a.selectbox(
                "Sources", ["both", "commons", "openverse"],
                format_func=lambda v: {"both": "Wikimedia Commons + Openverse", "commons": "Wikimedia Commons", "openverse": "Openverse"}[v],
            )
            limit = b.select_slider("Candidate images", options=[40, 60, 80, 100, 120, 140, 160], value=120)
            submitted = st.form_submit_button("Collect images", type="primary", use_container_width=True)
        if submitted:
            try:
                with st.spinner("Searching and downloading the image cohort…"):
                    items, actual_query = cached_search(query, limit, provider)
                    progress = st.progress(0., text="Downloading images")
                    samples, failures = download_images(items, progress.progress)
                    progress.empty()
                if not samples:
                    st.error("No usable images were downloaded. Try another query or source.")
                else:
                    replace_dataset(samples, actual_query, failures)
                    st.rerun()
            except Exception as exc:
                st.error(f"Collection failed: {exc}")
    with upload_tab:
        files = st.file_uploader("Images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        if st.button("Use uploaded cohort", disabled=not files, type="primary"):
            samples, failures = [], []
            for file in (files or [])[:240]:
                try:
                    samples.append(decode_image(file.getvalue(), dict(title=file.name, provider="upload")))
                except Exception as exc:
                    failures.append(dict(title=file.name, error=str(exc)))
            if samples:
                replace_dataset(samples, "uploaded cohort", failures)
                st.rerun()
            else:
                st.error("No uploaded image could be decoded.")
    with restore_tab:
        archive = st.file_uploader("Modern Cliché experiment.zip", type=["zip"])
        if st.button("Restore experiment", disabled=archive is None, type="primary"):
            try:
                samples, overrides, manifest = load_experiment(archive.getvalue())
                replace_dataset(samples, manifest.get("query", "restored cohort"), overrides=overrides, manifest=manifest)
                st.rerun()
            except Exception as exc:
                st.error(f"Restore failed: {exc}")
    st.stop()

if "samples" not in st.session_state:
    st.session_state.step = 0
    st.rerun()

if st.session_state.step == 1:
    st.header("Curate automatically")
    st.markdown(
        '<div class="mc-note">Duplicates and weak foreground extractions are filtered automatically. '
        'Manual review is optional and limited to flagged cases.</div>', unsafe_allow_html=True,
    )
    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Downloaded", len(st.session_state.samples))
    m2.metric("Query", st.session_state.query)
    m3.metric("Download failures", len(st.session_state.download_failures))
    with st.expander("Curation settings", expanded="prepared" not in st.session_state):
        a, b, c = st.columns(3)
        method = a.selectbox(
            "Foreground extraction", ["u2net", "grabcut", "binary"],
            format_func=lambda v: {"u2net": "U²-Net", "grabcut": "GrabCut", "binary": "Binary silhouette"}[v],
        )
        threshold = b.slider("Mask threshold", .1, .9, .5, .05)
        size = c.select_slider("Canonical silhouette size", options=[64, 96, 128], value=96)
        if st.button("Run automatic curation", type="primary", use_container_width=True):
            try:
                with st.spinner("Removing duplicates and extracting foregrounds…"):
                    progress = st.progress(0.)
                    prepared = prepare(
                        st.session_state.samples, method, threshold, size,
                        overrides=st.session_state.get("overrides", {}), progress=progress.progress,
                    )
                    progress.empty()
                prepared["failures"].extend(st.session_state.download_failures)
                st.session_state.prepared = prepared
                st.session_state.inclusion = {
                    sample.id: default_include(info)
                    for sample, info in zip(prepared["samples"], prepared["quality"])
                }
                _clear_downstream(2)
                st.rerun()
            except Exception as exc:
                st.error(f"Curation failed: {exc}")
    if "prepared" not in st.session_state:
        st.image([sample.image for sample in st.session_state.samples[:12]], width=150)
        if st.button("← Back", use_container_width=True):
            go(0)
        st.stop()

    prepared = st.session_state.prepared
    scores = [quality_score(info) for info in prepared["quality"]]
    included = [i for i, sample in enumerate(prepared["samples"]) if st.session_state.inclusion.get(sample.id, False)]
    flagged = [i for i in range(len(prepared["samples"])) if i not in included]
    metrics = st.columns(4)
    metrics[0].metric("Unique objects", len(prepared["samples"]))
    metrics[1].metric("Auto accepted", len(included))
    metrics[2].metric("Flagged / rejected", len(flagged))
    metrics[3].metric("Duplicates removed", len(prepared["duplicates"]))
    if prepared["samples"]:
        order = np.argsort(scores)[::-1][:12]
        st.caption("Highest-confidence foregrounds")
        st.image([overlay(prepared["samples"][i].image, prepared["raw_masks"][i]) for i in order], width=145)
    if flagged:
        with st.expander(f"Optional review · {len(flagged)} flagged images"):
            pages = max(1, (len(flagged) + 11) // 12)
            page = st.selectbox("Page", list(range(pages)), format_func=lambda i: f"{i + 1} / {pages}") if pages > 1 else 0
            visible = flagged[page * 12:(page + 1) * 12]
            columns = st.columns(4)
            for slot, index in enumerate(visible):
                sample = prepared["samples"][index]
                with columns[slot % 4]:
                    st.image(overlay(sample.image, prepared["raw_masks"][index]), width="stretch")
                    selected = st.checkbox(
                        f"Keep · score {scores[index]:.2f}",
                        value=st.session_state.inclusion.get(sample.id, False), key=f"include_{sample.id}",
                    )
                    st.session_state.inclusion[sample.id] = selected
                    st.caption(sample.metadata.get("title", sample.id)[:80])
    included = [i for i, sample in enumerate(prepared["samples"]) if st.session_state.inclusion.get(sample.id, False)]
    st.divider()
    left, right = st.columns(2)
    if left.button("← Back to collection", use_container_width=True):
        go(0)
    if right.button("Continue to discovery →", type="primary", disabled=len(included) < 8, use_container_width=True):
        st.session_state.curation_indices = included
        _clear_downstream(3)
        go(2)
    if len(included) < 8:
        st.caption("Keep at least 8 images. 30–100 coherent samples work much better for automatic discovery.")
    st.stop()

if st.session_state.step == 2:
    if "prepared" not in st.session_state or "curation_indices" not in st.session_state:
        go(1)
    prepared = st.session_state.prepared
    indices = st.session_state.curation_indices
    used = subset(prepared, indices)
    st.header("Discover morphology groups")
    st.markdown(
        '<div class="mc-note">The default path combines DINO visual embeddings with silhouette geometry. '
        'HDBSCAN discovers dense groups and leaves ambiguous samples as outliers instead of forcing every image into a category.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    backend = "dino" if ml_available() else "shape"
    with st.expander("Discovery settings"):
        a, b = st.columns(2)
        backend = a.selectbox(
            "Feature system", ["dino", "shape"] if ml_available() else ["shape"],
            format_func=lambda v: "DINO + geometry" if v == "dino" else "Geometry only",
        )
        requested_k = b.selectbox(
            "Grouping", [0, 2, 3, 4, 5, 6],
            format_func=lambda v: "Automatic density discovery" if v == 0 else f"Manual K-Means · {v} groups",
        )
        seed = st.number_input("Seed", 0, 999999, int(st.session_state.seed))
    signature = (fingerprint(used, list(range(len(used["samples"])))), backend, int(seed), int(requested_k))
    stale = st.session_state.get("analysis_signature") != signature
    if st.button("Discover groups", type="primary", use_container_width=True) or ("analysis" not in st.session_state):
        try:
            with st.spinner("Extracting visual features and discovering stable groups…"):
                embeddings = None
                actual_backend = backend
                if backend == "dino":
                    try:
                        embeddings, _ = visual_embeddings(
                            [sample.image for sample in used["samples"]], used["raw_masks"], "dino", st.session_state.query
                        )
                    except Exception as exc:
                        st.warning(f"DINO could not load; continuing with geometry only. {exc}")
                        actual_backend = "shape"
                result = analyse(used["masks"], int(seed), int(requested_k), embeddings, actual_backend)
                result["requested_k"] = int(requested_k)
                st.session_state.update(analysis=result, used=used, analysis_signature=signature)
                st.session_state.pop("selected_group", None)
                st.session_state.pop("mesh", None)
                st.session_state.pop("aggregate", None)
                st.rerun()
        except Exception as exc:
            st.error(f"Discovery failed: {exc}")
            st.stop()
    if "analysis" not in st.session_state:
        st.stop()
    if stale and st.session_state.get("analysis_signature") != signature:
        st.info("Settings changed. Run discovery again to update the groups.")
        st.stop()

    analysis, used = st.session_state.analysis, st.session_state.used
    outlier_count = int((analysis["labels"] < 0).sum())
    summary = st.columns(4)
    summary[0].metric("Analysed", len(used["samples"]))
    summary[1].metric("Morphology groups", len(analysis["groups"]))
    summary[2].metric("Outliers", outlier_count)
    summary[3].metric("Separation", "—" if analysis["clusters"]["silhouette"] is None else f"{analysis['clusters']['silhouette']:.2f}")
    st.caption(analysis["clusters"]["reason"])
    projection = pd.DataFrame({
        "PC1": analysis["projection"][:, 0], "PC2": analysis["projection"][:, 1],
        "group": ["OUTLIER" if value < 0 else f"GROUP {value + 1}" for value in analysis["labels"]],
        "title": [sample.metadata.get("title", sample.id) for sample in used["samples"]],
    })
    figure = px.scatter(projection, x="PC1", y="PC2", color="group", hover_data=["title"])
    figure.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0), font=dict(family=FONT), legend_title_text="")
    st.plotly_chart(figure, width="stretch")
    st.subheader("Discovered groups")
    group_columns = st.columns(min(3, max(1, len(analysis["groups"]))))
    for group_index, group in enumerate(analysis["groups"]):
        with group_columns[group_index % len(group_columns)]:
            medoid = group["medoid"]
            st.image(used["samples"][medoid].image, width="stretch")
            st.markdown(f"**GROUP {group_index + 1}** · {len(group['indices'])} images")
            if group["descriptive_rules"]:
                rules = ", ".join(rule["label"] for rule in group["descriptive_rules"][:2])
                st.caption(f"Strongest geometric differences: {rules}")
            nearest = sorted(group["indices"], key=lambda i: analysis["distance"][i])[:5]
            st.image([used["samples"][i].image for i in nearest], width=72)
    group_choice = st.selectbox(
        "Morphology to reconstruct", list(range(len(analysis["groups"]))),
        index=int(st.session_state.get("selected_group", 0)),
        format_func=lambda i: f"Group {i + 1} · {len(analysis['groups'][i]['indices'])} images",
    )
    st.divider()
    left, right = st.columns(2)
    if left.button("← Back to curation", use_container_width=True):
        go(1)
    if right.button("Continue to reconstruction →", type="primary", use_container_width=True):
        st.session_state.selected_group = int(group_choice)
        _clear_downstream(4)
        go(3)
    st.stop()

if st.session_state.step == 3:
    if "analysis" not in st.session_state:
        go(2)
    analysis, used = st.session_state.analysis, st.session_state.used
    group_index = int(st.session_state.get("selected_group", 0))
    group = analysis["groups"][group_index]
    members = sorted(group["indices"], key=lambda i: analysis["distance"][i])
    st.header("Reconstruct the statistical visible surface")
    st.markdown(
        '<div class="mc-note">Only the most central members of the selected cluster receive monocular depth inference. '
        'Their point maps are scale/translation-normalised, fitted to a shared canonical frame, and aggregated by median depth. '
        'Coverage shows repeated structure; variation shows where the cluster disagrees. Hidden backsides remain unobserved.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.image([used["samples"][i].image for i in members[:8]], width=120)
    max_specimens = min(12, len(members))
    a, b, c = st.columns(3)
    specimen_count = a.slider("Depth specimens", min_value=2, max_value=max_specimens, value=min(max_specimens, 6)) if max_specimens >= 2 else 1
    depth_options = (["moge-small", "moge-base"] if moge_available() else []) + ["depth-anything"]
    depth_model = b.selectbox("Depth model", depth_options, index=0)
    depth_size = c.select_slider("Inference resolution", options=[384, 512, 768], value=512)
    d, e, f = st.columns(3)
    coverage_threshold = d.slider("Core coverage", .25, .80, .45, .05)
    edge_threshold = e.slider("Surface continuity", .03, .20, .10, .01)
    size_mm = f.number_input("Export size · mm", min_value=20., max_value=1000., value=160., step=10.)
    detail = 7 if depth_model.startswith("moge") else 0
    if st.button("Build statistical surface", type="primary", disabled=max_specimens < 2, use_container_width=True):
        selected = members[:specimen_count]
        results = []
        progress = st.progress(0., text="Estimating visible 3D surfaces")
        try:
            for position, index in enumerate(selected, 1):
                sample = used["samples"][index]
                key = (sample.id, depth_model, int(depth_size), int(detail))
                if key not in st.session_state.depth_cache:
                    st.session_state.depth_cache[key] = infer_depth(
                        sample.image, used["raw_masks"][index], depth_model, int(depth_size), int(detail)
                    )
                results.append(st.session_state.depth_cache[key])
                progress.progress(position / len(selected), text=f"Depth {position} / {len(selected)}")
            progress.progress(1., text="Aggregating canonical point maps")
            aggregate = aggregate_depth_results(results, size=160, coverage_threshold=float(coverage_threshold))
            mesh, geometry = surface_mesh(
                aggregate, size_mm=float(size_mm), edge_threshold=float(edge_threshold), max_resolution=256, thickness_mm=0.
            )
            recipe = dict(
                mode="statistical-visible-surface", group_index=group_index,
                specimen_count=len(selected), input_indices=[int(i) for i in selected],
                depth_model=depth_model, depth_size=int(depth_size),
                coverage_threshold=float(coverage_threshold), aggregation="canonical median point-map surface",
            )
            st.session_state.update(aggregate=aggregate, mesh=mesh, geometry=geometry, recipe=recipe)
            st.session_state.pop("ready_zip", None)
            progress.empty()
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"Reconstruction failed: {exc}")
    if "mesh" in st.session_state:
        aggregate, mesh, geometry = st.session_state.aggregate, st.session_state.mesh, st.session_state.geometry
        previews = st.columns([1, 1, 2])
        previews[0].image(coverage_preview(aggregate), caption="COVERAGE", width="stretch")
        previews[1].image(variation_preview(aggregate), caption="DEPTH VARIATION", width="stretch")
        with previews[2]:
            st.plotly_chart(figure_mesh(mesh), width="stretch")
        stats = st.columns(4)
        stats[0].metric("Specimens", aggregate["metadata"]["specimens"])
        stats[1].metric("Surface vertices", f"{len(mesh.vertices):,}")
        stats[2].metric("Triangles", f"{len(mesh.faces):,}")
        stats[3].metric("Components", geometry["components_3d"])
        st.caption("This mesh represents repeated camera-facing structure inside the selected cluster. It is not a complete volumetric object.")
    st.divider()
    left, right = st.columns(2)
    if left.button("← Back to groups", use_container_width=True):
        go(2)
    if right.button("Continue to export →", type="primary", disabled="mesh" not in st.session_state, use_container_width=True):
        go(4)
    st.stop()

if st.session_state.step == 4:
    if "mesh" not in st.session_state:
        go(3)
    analysis, used = st.session_state.analysis, st.session_state.used
    mesh, geometry, recipe = st.session_state.mesh, st.session_state.geometry, st.session_state.recipe
    aggregate = st.session_state.aggregate
    st.header("Export the result and the evidence")
    st.markdown(
        '<div class="mc-note">Export the mesh, or save the full experiment with source images, masks, clustering data, '
        'the statistical surface and provenance.</div>', unsafe_allow_html=True,
    )
    st.write("")
    st.plotly_chart(figure_mesh(mesh), width="stretch")
    columns = st.columns(4)
    for column, ext in zip(columns, ["stl", "obj", "glb", "ply"]):
        column.download_button(
            ext.upper(), mesh_bytes(mesh, ext), f"modern_cliche.{ext}",
            mime={"stl": "application/octet-stream", "obj": "text/plain", "glb": "model/gltf-binary", "ply": "application/octet-stream"}[ext],
            use_container_width=True,
        )
    st.write("")
    export_signature = dumps(dict(recipe=recipe, geometry=geometry))
    if st.button("Prepare experiment.zip", type="primary", use_container_width=True):
        try:
            with st.spinner("Packing source images, masks, analysis and statistical 3D data…"):
                raw = experiment_zip(
                    used, analysis, st.session_state.query, mesh, recipe, geometry,
                    generated_mask=aggregate["valid"], selected_depth=aggregate,
                )
                st.session_state.ready_zip = (export_signature, raw)
        except Exception as exc:
            st.error(f"Export package failed: {exc}")
    ready = st.session_state.get("ready_zip")
    if ready and ready[0] == export_signature:
        st.download_button(
            "Download experiment.zip", ready[1], "experiment.zip", "application/zip",
            type="primary", use_container_width=True,
        )
    with st.expander("Method record"):
        st.json({
            "query": st.session_state.query, "feature_backend": analysis["backend"],
            "cluster_reason": analysis["clusters"]["reason"], "recipe": recipe, "geometry": geometry,
        })
    st.divider()
    left, right = st.columns(2)
    if left.button("← Back to reconstruction", use_container_width=True):
        go(3)
    if right.button("Start a new cohort", use_container_width=True):
        for key in list(st.session_state):
            if key not in ("seed", "depth_cache"):
                del st.session_state[key]
        st.session_state.step = 0
        st.rerun()
