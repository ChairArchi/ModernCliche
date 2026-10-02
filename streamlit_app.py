from __future__ import annotations

from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modern_cliche.data import search_images, download_images, decode_image
from modern_cliche.segmentation import overlay
from modern_cliche.pipeline import prepare, subset, fingerprint, load_experiment
from modern_cliche.analysis import analyse, synthesize_field, ml_available, visual_embeddings
from modern_cliche.geometry import lift_to_mesh
from modern_cliche.depth import infer_cohort, infer_depth, moge_available, depth_preview, encode_depth
from modern_cliche.surface import surface_mesh
from modern_cliche.export import experiment_zip, sample_table, png_bytes, dumps, mesh_bytes

st.set_page_config(page_title="Modern Cliché — Depth & Shape Research", layout="wide")
st.title("MODERN CLICHÉ")
st.caption("대상 검색 → 이미지 수집 → 분류 → 형상 규칙 탐색 → 3D 모델화")
st.write("수집한 대상의 형상과 추정 깊이를 분석하고, 측정한 차이를 3D 표면으로 연결하는 연구 도구입니다.")


def replace_dataset(samples, query, failures=None, overrides=None, manifest=None):
    for key in list(st.session_state):
        if key.startswith(("include_", "mask_edit_", "generation_")) or key in ("prepared", "analysis", "used", "analysis_signature", "individual_depth", "ready_zip"):
            del st.session_state[key]
    st.session_state.update(samples=samples, query=query, download_failures=failures or [],
                            overrides=overrides or {}, restored=manifest or {}, inclusion={})


@st.cache_data(ttl=1800, max_entries=12, show_spinner=False)
def cached_search(query, limit, provider):
    return search_images(query, limit, provider)


@st.cache_data(max_entries=4, show_spinner=False)
def cached_mesh(mask, depth, size):
    return lift_to_mesh(mask, depth, size)


@st.cache_data(max_entries=3, show_spinner=False)
def cached_surface(result, size, edge, resolution, thickness):
    return surface_mesh(result, size, edge, resolution, thickness)


with st.sidebar:
    st.write("**실험 설정**")
    seed = st.number_input("반복 실험 시드", 0, 999999, 17)
    method = st.selectbox("대상 영역 추출", ["u2net", "grabcut", "binary"],
                          format_func=lambda v: {"u2net": "U²-Net 자동 배경 분리", "grabcut": "GrabCut 중앙 대상 추정", "binary": "검은 실루엣 추출"}[v])
    threshold = st.slider("마스크 경계 기준", .1, .9, .5, .05)
    size = st.select_slider("분석 해상도", options=[64, 96, 128], value=96)
    backend_options = ["shape"] + (["dino", "clip"] if ml_available() else [])
    backend = st.selectbox("분류에 사용할 특징", backend_options,
                           format_func=lambda v: {"shape": "형태 통계 + 거리장", "dino": "DINOv2 대상 시각 특징", "clip": "CLIP 대상 시각 특징"}[v])
    st.caption("CPU 실행을 지원합니다. DINOv2와 CLIP 특징도 비교할 수 있습니다.")
    depth_options = (["moge-base", "moge-small", "moge-large"] if moge_available() else []) + ["depth-anything"]
    depth_model = st.selectbox("깊이 추정 모델", depth_options,
        format_func=lambda v: {"moge-small": "MoGe-2 Small · 카메라와 표면 추정", "moge-base": "MoGe-2 Base · 정밀 비교", "moge-large": "MoGe-2 Large · 높은 메모리 사용",
                               "depth-anything": "Depth Anything V2 · 상대 깊이 비교"}[v])
    depth_size = st.select_slider("깊이 맵 최대 해상도", options=[384, 512, 768], value=768)
    detail = st.slider("MoGe 내부 추론 정밀도", 0, 9, 9, disabled=depth_model == "depth-anything")
    st.caption("깊이 추정은 Python 내부에서 실행됩니다. 최초 실행 시 모델을 다운로드하며, 동일 조건의 결과는 재사용합니다.")
    if not moge_available():
        st.info("MoGe-2 실행에는 최신 requirements.txt 설치가 필요합니다.")
    requested_k = st.selectbox("그룹 수", [0, 2, 3, 4, 5, 6], format_func=lambda v: "자동 탐색" if v == 0 else str(v))
    st.caption("그룹은 데이터의 탐색적 구분입니다. 대상의 보편 법칙이나 신체 부위를 의미하지 않습니다.")
    with st.expander("방법과 참고 문헌"):
        st.markdown(Path("docs/REFERENCES.md").read_text(encoding="utf-8"))

st.subheader("1 · 이미지 수집")
search_tab, upload_tab, restore_tab, demo_tab = st.tabs(["대상 검색", "직접 업로드", "실험 ZIP 복원", "검증용 데이터"])
with search_tab:
    with st.form("search_form"):
        query = st.text_input("대상", "가고일", help="가고일·개·고양이·새·문은 영어 검색어로 변환됩니다. 다른 대상은 입력한 검색어를 사용합니다.")
        col_a, col_b = st.columns(2)
        provider = col_a.selectbox("이미지 출처", ["commons", "openverse"], format_func=lambda v: "Wikimedia Commons" if v == "commons" else "Openverse")
        limit = col_b.slider("수집 후보 수", 8, 80, 32, 4)
        submitted = st.form_submit_button("검색하고 이미지 수집", type="primary")
    if submitted:
        try:
            with st.spinner("이미지 후보를 검색합니다…"):
                items, actual_query = cached_search(query, limit, provider)
                progress = st.progress(0., text="이미지 다운로드")
                samples, failures = download_images(items, progress.progress)
                progress.empty()
            if not samples:
                st.error("수집된 이미지가 없습니다. 출처를 변경하거나 이미지를 업로드하십시오.")
                if failures:
                    st.dataframe(pd.DataFrame(failures), hide_index=True)
            else:
                replace_dataset(samples, actual_query, failures)
                st.success(f"{len(samples)}장을 수집했습니다. 다음 단계에서 검색 결과와 대상의 일치 여부를 확인하십시오.")
        except Exception as exc:
            st.error(f"검색에 실패했습니다: {exc}")
with upload_tab:
    files = st.file_uploader("사진 또는 투명 배경 PNG", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    label_file = st.file_uploader("분류 라벨 CSV · 선택 사항", type=["csv"], help="filename,label 두 열을 사용합니다. 기존 데이터의 분류 라벨을 보존할 수 있습니다.")
    if st.button("업로드한 이미지 사용", disabled=not files):
        samples, failures = [], []
        try:
            label_map = dict(pd.read_csv(label_file)[["filename", "label"]].itertuples(index=False, name=None)) if label_file else {}
            for file in files[:80]:
                try:
                    samples.append(decode_image(file.getvalue(), dict(title=file.name, provider="upload",
                                                dataset_label=str(label_map.get(file.name, "")))))
                except Exception as exc:
                    failures.append(dict(title=file.name, error=str(exc)))
            if samples:
                replace_dataset(samples, "uploaded dataset", failures)
        except Exception as exc:
            st.error(f"업로드 파일을 읽지 못했습니다: {exc}")
with restore_tab:
    uploaded_zip = st.file_uploader("이 프로그램이 저장한 experiment.zip", type=["zip"])
    if st.button("저장한 데이터와 마스크 복원", disabled=uploaded_zip is None):
        try:
            samples, overrides, manifest = load_experiment(uploaded_zip.getvalue())
            replace_dataset(samples, manifest.get("query", ""), overrides=overrides, manifest=manifest)
            st.success("원본과 마스크를 복원했습니다. 이전 분석 설정과 생성값을 아래에서 확인할 수 있습니다.")
        except Exception as exc:
            st.error(f"실험 파일을 읽지 못했습니다: {exc}")
with demo_tab:
    st.caption("타원·십자로 구성된 파이프라인 검증용 합성 실루엣입니다. 실제 이미지 및 깊이 모델 검증과 구분됩니다.")
    if st.button("검증용 16장 불러오기"):
        from modern_cliche.demo import demo_samples
        replace_dataset(demo_samples(), "synthetic verification")

if "samples" not in st.session_state:
    st.info("검색하거나 이미지를 업로드하면 수집·분류·형상 비교를 시작할 수 있습니다.")
    st.stop()
st.caption(f"현재 데이터: {st.session_state.query} · {len(st.session_state.samples)}장")
if st.session_state.download_failures:
    with st.expander(f"다운로드/업로드 실패 {len(st.session_state.download_failures)}건"):
        st.dataframe(pd.DataFrame(st.session_state.download_failures), hide_index=True)
if st.session_state.restored:
    with st.expander("이전 실험 설정"):
        st.json({key: st.session_state.restored.get(key) for key in ("preparation", "feature_backend", "requested_k", "seed", "generation", "geometry")})

st.subheader("2 · 대상 추출과 데이터 검토")
st.caption("배경·가림·다중 대상에 의해 추출 오류가 발생할 수 있습니다. 주황 영역을 검토하고 잘못된 이미지를 제외하십시오.")
if st.button("대상 영역 추출", type="primary"):
    with st.spinner("대상 영역을 추출합니다. 첫 실행 시 배경 분리 모델을 다운로드합니다…"):
        progress = st.progress(0.)
        prepared = prepare(st.session_state.samples, method, threshold, size,
                           overrides=st.session_state.overrides, progress=progress.progress)
        prepared["failures"].extend(st.session_state.download_failures)
        st.session_state.prepared = prepared
        st.session_state.inclusion = {sample.id: st.session_state.inclusion.get(sample.id, not quality.get("needs_review", False))
                                      for sample, quality in zip(prepared["samples"], prepared["quality"])}
        for key in ("analysis", "used", "analysis_signature"):
            st.session_state.pop(key, None)
        progress.empty()
if "prepared" not in st.session_state:
    st.image([sample.image for sample in st.session_state.samples[:12]], width=160)
    st.stop()
prepared = st.session_state.prepared
if prepared["duplicates"] or prepared["failures"]:
    with st.expander(f"제외 기록 · 중복 {len(prepared['duplicates'])} / 실패 {len(prepared['failures'])}"):
        st.json(dict(duplicates=prepared["duplicates"], failures=prepared["failures"]))
if not prepared["samples"]:
    st.error("추출된 대상이 없습니다. 실패 기록을 확인하고 다른 분리 방법이나 투명 PNG를 사용하십시오.")
    st.stop()
page = st.selectbox("이미지 검토 페이지", list(range((len(prepared["samples"]) + 11) // 12)), format_func=lambda i: str(i + 1))
columns = st.columns(4)
for index in range(page * 12, min((page + 1) * 12, len(prepared["samples"]))):
    sample, quality = prepared["samples"][index], prepared["quality"][index]
    with columns[index % 4]:
        st.image(overlay(sample.image, prepared["raw_masks"][index]), width="stretch")
        included = st.checkbox(f"{index + 1} · 분석에 포함", value=st.session_state.inclusion[sample.id], key=f"include_{sample.id}")
        st.session_state.inclusion[sample.id] = included
        st.caption(sample.metadata.get("title", sample.id)[:100])
        if quality.get("needs_review"):
            st.caption("경계 접촉 또는 분리된 영역이 커서 검토가 필요합니다.")
        url = sample.metadata.get("source_url", "")
        if url.startswith("https://"):
            st.markdown(f"[출처]({url}) · {sample.metadata.get('license', '')}")
        with st.expander("마스크와 추출 정보"):
            st.image(prepared["masks"][index].astype(np.uint8) * 255, width=160)
            st.json(quality)
            st.download_button("원본 PNG", png_bytes(sample.image), f"{sample.id}.png", "image/png", key=f"raw_{sample.id}")
            manual = st.file_uploader("같은 크기의 수정 마스크 · 흰색이 대상", type=["png"], key=f"mask_edit_{sample.id}")
            if manual and st.button("수정 마스크 적용", key=f"apply_{sample.id}"):
                try:
                    with Image.open(BytesIO(manual.getvalue())) as im:
                        mask = np.asarray(im.convert("L")) > 127
                    if mask.shape != prepared["raw_masks"][index].shape:
                        raise ValueError("원본 PNG와 가로·세로 크기가 같아야 합니다.")
                    st.session_state.overrides[sample.id] = mask
                    st.info("수정 마스크를 저장했습니다. ‘대상 영역 추출’을 다시 실행하면 반영됩니다.")
                except Exception as exc:
                    st.error(str(exc))

indices = [i for i, sample in enumerate(prepared["samples"])
           if st.session_state.inclusion[sample.id]]
st.caption(f"분석 선택: {len(indices)}장. 권장 20장 이상 · 최소 3장. 해상도 및 추출 설정은 다시 추출해야 적용됩니다.")
include_depth = st.checkbox("깊이 특징을 분류와 규칙 분석에 포함", value=st.session_state.query != "synthetic verification")
st.caption("깊이 범위·기울기·불연속을 함께 비교합니다. 방향과 자세가 다른 이미지는 깊이 통계도 달라질 수 있습니다.")
signature = (fingerprint(prepared, indices), backend, int(seed), requested_k, include_depth, depth_model, depth_size, detail)
st.subheader("3 · Sorting과 형상 규칙 탐색")
if st.button("선택한 데이터 분석", disabled=len(indices) < 3, type="primary"):
    try:
        with st.spinner("형상 특징·깊이·분류 안정성·변형 방향을 계산합니다…"):
            used = subset(prepared, indices)
            used["excluded_samples"] = [dict(id=s.id, metadata=s.metadata, reason="manual-or-review-exclusion")
                                         for i, s in enumerate(prepared["samples"]) if i not in indices]
            embeddings, semantic = None, None
            if backend != "shape":
                embeddings, semantic = visual_embeddings([s.image for s in used["samples"]], used["raw_masks"], backend, st.session_state.query)
            if include_depth:
                progress = st.progress(0., text="이미지별 깊이 추정")
                saved = st.session_state.restored.get("_saved_depth", {})
                saved_settings = st.session_state.restored.get("depth_estimation") or {}
                settings_match = (saved_settings.get("model") == depth_model and saved_settings.get("max_size") == depth_size
                                  and saved_settings.get("resolution_level") == detail)
                if saved and settings_match and all(sample.id in saved for sample in used["samples"]) and st.session_state.restored.get("fingerprint") == fingerprint(used, list(range(len(used["samples"])))):
                    used["depth_results"] = [saved[sample.id] for sample in used["samples"]]
                else:
                    used["depth_results"] = infer_cohort(used, depth_model, depth_size, detail, progress=progress.progress)
                progress.empty()
            result = analyse(used["masks"], int(seed), requested_k, embeddings, backend, used.get("depth_results"))
            result["requested_k"] = requested_k
            if semantic is not None:
                for s, score in zip(used["samples"], semantic):
                    s.metadata["clip_query_cosine"] = float(score)
            st.session_state.update(analysis=result, used=used, analysis_signature=signature)
    except Exception as exc:
        st.error(f"분석에 실패했습니다: {exc}")
if "analysis" not in st.session_state:
    st.stop()
if signature != st.session_state.analysis_signature:
    st.warning("포함 이미지 또는 분석 설정이 변경되었습니다. ‘선택한 데이터 분석’을 다시 실행하십시오.")
    st.stop()
analysis, used = st.session_state.analysis, st.session_state.used
table = sample_table(used, analysis)
summary = st.columns(4)
summary[0].metric("분석 이미지", len(used["samples"]))
summary[1].metric("형상 그룹", len(analysis["groups"]))
summary[2].metric("그룹 분리도", "—" if analysis["clusters"]["silhouette"] is None else f"{analysis['clusters']['silhouette']:.3f}")
summary[3].metric("반복 안정성 · ARI", "—" if analysis["clusters"]["stability"] is None else f"{analysis['clusters']['stability']:.3f}")
if len(used["samples"]) < 12:
    st.warning("표본 수가 작습니다. 결과의 해석 범위를 이 데이터로 제한하십시오.")
st.caption(analysis["clusters"]["reason"] + " · 군집 평가는 의미 분류의 정확성이나 인간의 인식 점수가 아닙니다.")
sorting_tab, rules_tab, axes_tab = st.tabs(["분포와 Sorting", "그룹별 형상 차이", "데이터에서 계산한 변형 방향"])
with sorting_tab:
    projection = table[["title", "group", "distance"]].copy()
    projection["PC1"], projection["PC2"] = analysis["projection"].T
    projection["group"] = projection["group"].astype(str)
    st.plotly_chart(px.scatter(projection, x="PC1", y="PC2", color="group", hover_data=["title", "distance"]), width="stretch")
    st.caption("2D PCA 지도는 시각화용입니다. 분류는 투영 전 특징 공간에서 수행합니다.")
    sort = st.selectbox("정렬 기준", ["group", "distance", "silhouette", "aspect_ratio", "occupancy"])
    st.dataframe(table.sort_values(sort), hide_index=True)
    labels = table["dataset_label"].fillna("")
    if (labels != "").any():
        st.write("**원래 데이터 라벨과 형상 그룹 비교**")
        st.dataframe(pd.crosstab(labels, table["group"]))
    if analysis["clusters"]["candidates"]:
        st.dataframe(pd.DataFrame(analysis["clusters"]["candidates"]), hide_index=True)
with rules_tab:
    for group in analysis["groups"]:
        st.write(f"**그룹 {group['label']} · {len(group['indices'])}장**")
        left, right = st.columns([1, 3])
        left.image(used["samples"][group["medoid"]].image, caption="특징 거리로 선택한 대표 이미지", width=180)
        right.dataframe(pd.DataFrame(group["descriptive_rules"]), hide_index=True)
        if group.get("depth_rules"):
            right.write("**추정 깊이에서 측정한 차이**")
            right.dataframe(pd.DataFrame(group["depth_rules"]), hide_index=True)
        st.image([mask.astype(np.uint8) * 255 for i in group["indices"] for mask in [used["masks"][i]]], width=96)
    st.caption("standardised_difference는 전체 데이터 대비 평균 차이입니다. 신체 부위나 인과관계를 판별하는 값이 아닙니다.")
with axes_tab:
    model = analysis["model"]
    st.caption(f"최대 12개 변형 방향으로 전체 거리장 변동의 {model['retained_variance']:.1%}를 설명합니다. 방향·자세 차이도 변형에 포함됩니다.")
    for axis in range(min(4, len(model["components"]))):
        st.write(f"**변형 방향 {axis + 1} · 설명 분산 {model['explained'][axis]:.1%}**")
        masks = [(model["mean"] + value * model["std"][axis] * model["components"][axis]) > 0 for value in [-2, 0, 2]]
        st.image([mask.astype(np.uint8) * 255 for mask in masks], caption=["−2σ", "평균", "+2σ"], width=160)
        st.dataframe(pd.DataFrame(analysis["correlations"][axis]), hide_index=True)
    st.caption("상관은 이 표본에서 함께 변한 특징을 나타냅니다. 대상의 보편 법칙을 의미하지 않습니다.")
    st.write("**미사용 이미지 재구성 비교 · 반복 홀드아웃**")
    validation = analysis["validation"]
    if validation["available"]:
        st.dataframe(pd.DataFrame(validation["repeats"]), hide_index=True)
        st.caption(validation["interpretation"])
    else:
        st.caption(validation["reason"])

st.subheader("4 · 깊이 표면과 3D 모델")
st.caption("이미지에서 보이는 표면을 추정합니다. 보이지 않는 뒷면은 관측되지 않으며, 추가 두께는 제작용 가정으로 기록됩니다.")
path = st.radio("3D 구성 방법", ["depth", "silhouette"], index=1 if st.session_state.query == "synthetic verification" else 0,
                format_func=lambda v: "이미지 깊이 기반 표면" if v == "depth" else "실루엣 통계 변형 · 기존 반경 가정", horizontal=True)
controls, view = st.columns([1, 3])
mesh = geometry = recipe = generated = None
try:
    with controls:
        a = st.selectbox("기준 그룹", range(len(analysis["groups"])), format_func=lambda v: f"그룹 {v}")
        options = analysis["groups"][a]["indices"]
        medoid = analysis["groups"][a]["medoid"]
        size_mm = st.number_input("최대 크기 · mm", min_value=10., max_value=1000., value=120., step=10.)
        if path == "depth":
            specimen = st.selectbox("관측 이미지", options, index=options.index(medoid),
                format_func=lambda v: f"{v + 1} · {used['samples'][v].metadata.get('title', '')[:40]}")
            edge = st.slider("깊이 불연속 연결 기준", .005, .15, .03, .005,
                             help="상대 깊이 차이가 이 기준보다 큰 삼각형을 제거합니다. 기준을 높이면 가림 경계가 연결될 수 있습니다.")
            mesh_resolution = st.select_slider("메시 최대 해상도", options=[128, 256, 384, 512, 768], value=768)
            thickness = st.number_input("추가 두께 · mm · 0은 열린 표면", min_value=0., max_value=float(size_mm / 4 - .1), value=0., step=.5)
            depth_result = used.get("depth_results", [None] * len(used["samples"]))[specimen]
            if depth_result is None:
                key = (signature, specimen)
                individual = st.session_state.setdefault("individual_depth", {})
                if st.button("선택 이미지 깊이 추정", type="primary"):
                    with st.spinner("선택 이미지의 깊이와 표면을 추정합니다…"):
                        sample = used["samples"][specimen]
                        individual[key] = infer_depth(sample.image, used["raw_masks"][specimen], depth_model, depth_size, detail)
                depth_result = individual.get(key)
            if depth_result is not None:
                mesh, geometry = cached_surface(depth_result, size_mm, edge, mesh_resolution, thickness)
                generated = used["masks"][specimen]
                recipe = dict(mode="observed", geometry="depth", specimen=int(specimen), input_indices=[int(specimen)])
                st.caption("카메라: " + depth_result["metadata"]["camera"])
                st.download_button("추정 좌표·깊이 NPZ", encode_depth(depth_result), "depth.npz", "application/octet-stream")
            else:
                st.info("선택 이미지의 깊이를 추정하면 표면 모델을 생성합니다.")
        else:
            st.caption("이 경로는 실루엣 반경을 입체로 확장합니다. 이미지 깊이 추정 결과를 사용하지 않습니다.")
            mode = st.selectbox("형상 원천", ["observed", "variation", "interpolate"],
                format_func=lambda v: {"observed": "관측된 대표 실루엣", "variation": "그룹 안의 통계 변형", "interpolate": "그룹 사이 실루엣 보간"}[v])
            b, mix, coefficients, specimen = a, .5, [], None
            if mode == "observed":
                specimen = st.selectbox("관측 이미지", options, index=options.index(medoid),
                    format_func=lambda v: f"{v + 1} · {used['samples'][v].metadata.get('title', '')[:40]}")
            elif mode == "interpolate":
                b = st.selectbox("비교 그룹", range(len(analysis["groups"])), index=min(1, len(analysis["groups"]) - 1))
                mix = st.slider("비교 그룹 비율", 0., 1., .5, .05)
            else:
                local_model = analysis["groups"][a]["model"]
                for axis in range(min(4, len(local_model["components"]))):
                    coefficients.append(st.slider(f"변형 {axis + 1} · σ", -3., 3., 0., .1, key=f"generation_{a}_{axis}"))
            depth_scale = st.slider("깊이 가정 · 내부 반경 배율", .3, 2., 1., .1)
            generated, recipe = synthesize_field(analysis, mode, a, b, mix, coefficients, specimen)
            mesh, geometry = cached_mesh(generated, depth_scale, size_mm)
    if mesh is not None:
        with view:
            if path == "depth":
                left, right = st.columns(2)
                left.image(used["samples"][specimen].image, caption="분석 원본", width=220)
                right.image(depth_preview(depth_result), caption="추정 깊이 · 색상은 대상 내부에서 정규화", width=220)
                st.caption("깊이 표면은 단안 추정입니다. 실제 측정값이나 360° 복원이 아닙니다.")
            vertices, faces = mesh.vertices, mesh.faces
            display = st.selectbox("표면 표시", ["geometry", "source-color"], format_func=lambda v: "기하 형태" if v == "geometry" else "원본 색상") if path == "depth" else "geometry"
            colors = dict(vertexcolor=mesh.visual.vertex_colors) if display == "source-color" else dict(color="#b8b8b8")
            figure = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], **colors,
                flatshading=False, lighting=dict(ambient=.4, diffuse=.8)))
            figure.update_layout(height=540, margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(aspectmode="data", xaxis_title="X · mm", yaxis_title="추정 깊이 · mm", zaxis_title="높이 · mm", camera=dict(eye=dict(x=.8, y=-1.8, z=.6))))
            st.plotly_chart(figure, width="stretch")
            st.caption(f"닫힌 표면: {geometry['watertight']} · 분리된 표면: {geometry['components_3d']} · 삼각형: {geometry['faces']:,}")
            if path == "depth" and not geometry["watertight"]:
                st.info("열린 표면입니다. 제작용 닫힌 셸이 필요하면 추가 두께를 지정하십시오. 추가 면은 추정된 뒷면이 아닙니다.")
            if geometry["components_3d"] > 1:
                st.warning("분리된 표면이 포함되어 있습니다. 마스크와 깊이 경계를 확인하십시오.")
            with st.expander("생성 기록과 입체 측정"):
                st.json(dict(recipe=recipe, geometry=geometry))
        st.subheader("5 · 모델과 실험 기록 저장")
        cols = st.columns(5)
        for column, ext in zip(cols[:4], ["stl", "obj", "glb", "ply"]):
            column.download_button(ext.upper(), mesh_bytes(mesh, ext), f"modern_cliche.{ext}",
                mime={"stl": "application/octet-stream", "obj": "text/plain", "glb": "model/gltf-binary", "ply": "application/octet-stream"}[ext])
        export_signature = dumps(dict(analysis=signature, recipe=recipe, geometry=geometry))
        if cols[4].button("실험 ZIP 준비", type="primary"):
            with st.spinner("원본·깊이·분석·메시를 실험 ZIP으로 저장합니다…"):
                raw = experiment_zip(used, analysis, st.session_state.query, mesh, recipe, geometry, generated,
                    selected_depth=depth_result if path == "depth" else None)
                st.session_state.ready_zip = (export_signature, raw)
        ready = st.session_state.get("ready_zip")
        if ready and ready[0] == export_signature:
            cols[4].download_button("실험 전체 ZIP", ready[1], "experiment.zip", "application/zip")

except Exception as exc:
    st.error(f"3D 표면을 생성하지 못했습니다: {exc}")
    st.download_button("분석 기록 ZIP 저장", experiment_zip(used, analysis, st.session_state.query), "experiment.zip", "application/zip")
