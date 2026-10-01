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
from modern_cliche.analysis import analyse, synthesize_field, ml_available, visual_embeddings, FEATURE_LABELS
from modern_cliche.geometry import lift_to_mesh
from modern_cliche.export import experiment_zip, sample_table, png_bytes, dumps, mesh_bytes

st.set_page_config(page_title="Modern Cliché — Shape Research", layout="wide")
st.title("MODERN CLICHÉ")
st.caption("대상 검색 → 이미지 수집 → 분류 → 형상 규칙 탐색 → 3D 모델화")
st.write("수집한 대상의 공통 형상과 변형 범위를 계산하고, 그 계산을 입체로 연결하는 연구 도구야.")


def replace_dataset(samples, query, failures=None, overrides=None, manifest=None):
    for key in list(st.session_state):
        if key.startswith(("include_", "mask_edit_", "generation_")) or key in ("prepared", "analysis", "used", "analysis_signature"):
            del st.session_state[key]
    st.session_state.update(samples=samples, query=query, download_failures=failures or [],
                            overrides=overrides or {}, restored=manifest or {}, inclusion={})


@st.cache_data(ttl=1800, max_entries=12, show_spinner=False)
def cached_search(query, limit, provider):
    return search_images(query, limit, provider)


@st.cache_data(max_entries=4, show_spinner=False)
def cached_mesh(mask, depth, size):
    return lift_to_mesh(mask, depth, size)


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
    st.caption("기본 분석은 CPU로 실행돼. DINOv2/CLIP 비교는 requirements-ml.txt 설치 후 사용할 수 있어.")
    requested_k = st.selectbox("그룹 수", [0, 2, 3, 4, 5, 6], format_func=lambda v: "자동 탐색" if v == 0 else str(v))
    st.caption("그룹은 데이터의 탐색적 구분이야. 대상의 보편 법칙이나 의미 있는 신체 부위로 확정하지 않아.")
    with st.expander("방법과 참고 문헌"):
        st.markdown(Path("docs/REFERENCES.md").read_text(encoding="utf-8"))

st.subheader("1 · 이미지 수집")
search_tab, upload_tab, restore_tab, demo_tab = st.tabs(["대상 검색", "직접 업로드", "실험 ZIP 복원", "검증용 데이터"])
with search_tab:
    with st.form("search_form"):
        query = st.text_input("대상", "가고일", help="가고일·개·고양이·새·문은 영어 검색어로 연결돼. 다른 대상은 원하는 검색어 그대로 사용해.")
        col_a, col_b = st.columns(2)
        provider = col_a.selectbox("이미지 출처", ["commons", "openverse"], format_func=lambda v: "Wikimedia Commons" if v == "commons" else "Openverse")
        limit = col_b.slider("수집 후보 수", 8, 80, 32, 4)
        submitted = st.form_submit_button("검색하고 이미지 수집", type="primary")
    if submitted:
        try:
            with st.spinner("이미지 후보를 검색하고 있어…"):
                items, actual_query = cached_search(query, limit, provider)
                progress = st.progress(0., text="이미지 다운로드")
                samples, failures = download_images(items, progress.progress)
                progress.empty()
            if not samples:
                st.error("받아온 이미지가 없어. 출처를 바꾸거나 직접 이미지를 올려줘.")
                if failures:
                    st.dataframe(pd.DataFrame(failures), hide_index=True)
            else:
                replace_dataset(samples, actual_query, failures)
                st.success(f"{len(samples)}장 수집했어. 검색 결과가 대상과 맞는지는 다음 단계에서 확인해줘.")
        except Exception as exc:
            st.error(f"검색에 실패했어: {exc}")
with upload_tab:
    files = st.file_uploader("사진 또는 투명 배경 PNG", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    label_file = st.file_uploader("분류 라벨 CSV · 선택 사항", type=["csv"], help="filename,label 두 열. 캡차 등 이미 분류된 데이터의 라벨을 유지할 때 사용해.")
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
            st.error(f"업로드를 읽지 못했어: {exc}")
with restore_tab:
    uploaded_zip = st.file_uploader("이 프로그램이 저장한 experiment.zip", type=["zip"])
    if st.button("저장한 데이터와 마스크 복원", disabled=uploaded_zip is None):
        try:
            samples, overrides, manifest = load_experiment(uploaded_zip.getvalue())
            replace_dataset(samples, manifest.get("query", ""), overrides=overrides, manifest=manifest)
            st.success("원본과 마스크를 복원했어. 이전 분석 설정과 생성값은 아래에서 확인할 수 있어.")
        except Exception as exc:
            st.error(f"실험 파일을 읽지 못했어: {exc}")
with demo_tab:
    st.caption("파이프라인 검증용 타원·십자 합성 실루엣이야. 실제 대상의 수집 데이터나 디자인 제안이 아니야.")
    if st.button("검증용 16장 불러오기"):
        from modern_cliche.demo import demo_samples
        replace_dataset(demo_samples(), "synthetic verification")

if "samples" not in st.session_state:
    st.info("검색하거나 이미지를 올리면 수집·분류·형상 비교를 시작할 수 있어.")
    st.stop()
st.caption(f"현재 데이터: {st.session_state.query} · {len(st.session_state.samples)}장")
if st.session_state.download_failures:
    with st.expander(f"다운로드/업로드 실패 {len(st.session_state.download_failures)}건"):
        st.dataframe(pd.DataFrame(st.session_state.download_failures), hide_index=True)
if st.session_state.restored:
    with st.expander("이전 실험 설정"):
        st.json({key: st.session_state.restored.get(key) for key in ("preparation", "feature_backend", "requested_k", "seed", "generation", "geometry")})

st.subheader("2 · 대상 추출과 데이터 검토")
st.caption("배경·가림·여러 대상 때문에 추출이 틀릴 수 있어. 주황 영역이 실제 대상인지 확인하고, 잘못된 이미지는 제외해줘.")
if st.button("대상 영역 추출", type="primary"):
    with st.spinner("대상 영역을 추출하고 있어. 첫 실행은 작은 배경 분리 모델을 내려받아…"):
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
    st.error("추출된 대상이 없어. 실패 기록을 보고 다른 분리 방법이나 투명 PNG를 사용해줘.")
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
            st.caption("경계 접촉 또는 분리된 영역이 커서 검토가 필요해.")
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
                        raise ValueError("원본 PNG와 가로·세로 크기가 같아야 해.")
                    st.session_state.overrides[sample.id] = mask
                    st.info("수정 마스크를 저장했어. ‘대상 영역 추출’을 다시 누르면 반영돼.")
                except Exception as exc:
                    st.error(str(exc))

indices = [i for i, sample in enumerate(prepared["samples"])
           if st.session_state.inclusion[sample.id]]
st.caption(f"분석 선택: {len(indices)}장. 권장 20장 이상 · 최소 3장. 해상도/추출 설정 변경은 다시 추출해야 적용돼.")
signature = (fingerprint(prepared, indices), backend, int(seed), requested_k)
st.subheader("3 · Sorting과 형상 규칙 탐색")
if st.button("선택한 데이터 분석", disabled=len(indices) < 3, type="primary"):
    try:
        with st.spinner("형상 특징·분류 안정성·변형 방향을 계산하고 있어…"):
            used = subset(prepared, indices)
            used["excluded_samples"] = [dict(id=s.id, metadata=s.metadata, reason="manual-or-review-exclusion")
                                         for i, s in enumerate(prepared["samples"]) if i not in indices]
            embeddings, semantic = None, None
            if backend != "shape":
                embeddings, semantic = visual_embeddings([s.image for s in used["samples"]], used["raw_masks"], backend, st.session_state.query)
            result = analyse(used["masks"], int(seed), requested_k, embeddings, backend)
            result["requested_k"] = requested_k
            if semantic is not None:
                for s, score in zip(used["samples"], semantic):
                    s.metadata["clip_query_cosine"] = float(score)
            st.session_state.update(analysis=result, used=used, analysis_signature=signature)
    except Exception as exc:
        st.error(f"분석에 실패했어: {exc}")
if "analysis" not in st.session_state:
    st.stop()
if signature != st.session_state.analysis_signature:
    st.warning("포함 이미지 또는 분석 설정이 바뀌었어. ‘선택한 데이터 분석’을 다시 눌러줘.")
    st.stop()
analysis, used = st.session_state.analysis, st.session_state.used
table = sample_table(used, analysis)
summary = st.columns(4)
summary[0].metric("분석 이미지", len(used["samples"]))
summary[1].metric("형상 그룹", len(analysis["groups"]))
summary[2].metric("그룹 분리도", "—" if analysis["clusters"]["silhouette"] is None else f"{analysis['clusters']['silhouette']:.3f}")
summary[3].metric("반복 안정성 · ARI", "—" if analysis["clusters"]["stability"] is None else f"{analysis['clusters']['stability']:.3f}")
if len(used["samples"]) < 12:
    st.warning("표본이 작아. 지금 결과는 파이프라인 탐색용으로 읽어줘.")
st.caption(analysis["clusters"]["reason"] + " · 군집 평가는 의미의 정확성이나 인간의 인식 점수가 아니야.")
sorting_tab, rules_tab, axes_tab = st.tabs(["분포와 Sorting", "그룹별 형상 차이", "데이터에서 계산한 변형 방향"])
with sorting_tab:
    projection = table[["title", "group", "distance"]].copy()
    projection["PC1"], projection["PC2"] = analysis["projection"].T
    projection["group"] = projection["group"].astype(str)
    st.plotly_chart(px.scatter(projection, x="PC1", y="PC2", color="group", hover_data=["title", "distance"]), width="stretch")
    st.caption("2D PCA 지도는 표시용이야. 분류 계산은 투영 전 특징 공간에서 수행해.")
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
        st.image([mask.astype(np.uint8) * 255 for i in group["indices"] for mask in [used["masks"][i]]], width=96)
    st.caption("standardised_difference는 전체 데이터 대비 평균 차이야. 자동 발견한 ‘머리/날개’ 같은 의미나 인과관계가 아니야.")
with axes_tab:
    model = analysis["model"]
    st.caption(f"최대 12개 변형 방향으로 전체 거리장 변동의 {model['retained_variance']:.1%}를 담았어. 방향·자세 차이도 변형에 포함돼.")
    for axis in range(min(4, len(model["components"]))):
        st.write(f"**변형 방향 {axis + 1} · 설명 분산 {model['explained'][axis]:.1%}**")
        masks = [(model["mean"] + value * model["std"][axis] * model["components"][axis]) > 0 for value in [-2, 0, 2]]
        st.image([mask.astype(np.uint8) * 255 for mask in masks], caption=["−2σ", "평균", "+2σ"], width=160)
        st.dataframe(pd.DataFrame(analysis["correlations"][axis]), hide_index=True)
    st.caption("상관은 이 표본에서 같이 변한 특징이야. 대상의 보편 법칙으로 해석하지 않아.")
    st.write("**미사용 이미지 재구성 비교 · 반복 홀드아웃**")
    validation = analysis["validation"]
    if validation["available"]:
        st.dataframe(pd.DataFrame(validation["repeats"]), hide_index=True)
        st.caption(validation["interpretation"])
    else:
        st.caption(validation["reason"])

st.subheader("4 · 형상에서 3D로")
st.caption("깊이는 중심선의 내부 반경을 입체로 확장하는 가정이야. 실제 대상의 3D 복원과는 구분해서 읽어줘.")
controls, view = st.columns([1, 3])
with controls:
    mode = st.selectbox("형상 원천", ["observed", "variation", "interpolate"],
                        format_func=lambda v: {"observed": "관측된 대표 형상", "variation": "그룹 안의 통계 변형", "interpolate": "그룹 사이 형상 보간"}[v])
    a = st.selectbox("기준 그룹", range(len(analysis["groups"])), format_func=lambda v: f"그룹 {v}")
    b, mix, coefficients, specimen = a, .5, [], None
    if mode == "observed":
        options = analysis["groups"][a]["indices"]
        medoid = analysis["groups"][a]["medoid"]
        specimen = st.selectbox("관측 이미지", options, index=options.index(medoid),
                                format_func=lambda v: f"{v + 1} · {used['samples'][v].metadata.get('title', '')[:40]}")
    elif mode == "interpolate":
        b = st.selectbox("비교 그룹", range(len(analysis["groups"])), index=min(1, len(analysis["groups"]) - 1))
        mix = st.slider("비교 그룹 비율", 0., 1., .5, .05)
        if a == b:
            st.caption("같은 그룹을 골랐어. 평균 형상이 유지돼.")
    else:
        local_model = analysis["groups"][a]["model"]
        for axis in range(min(4, len(local_model["components"]))):
            coefficients.append(st.slider(f"변형 {axis + 1} · σ", -3., 3., 0., .1, key=f"generation_{a}_{axis}"))
        if not len(local_model["components"]):
            st.caption("이 그룹에는 계산 가능한 변형 방향이 없어.")
        if any(abs(v) > 2 for v in coefficients):
            st.caption("2σ 밖은 외삽 실험이야. 관측되지 않은 형상이 나올 수 있어.")
    depth = st.slider("깊이 가정 · 내부 반경 배율", .3, 2., 1., .1)
    size_mm = st.number_input("최대 길이 · mm", 10., 1000., 120., 10.)
try:
    generated, recipe = synthesize_field(analysis, mode, a, b, mix, coefficients, specimen)
    mesh, geometry = cached_mesh(generated, depth, size_mm)
    with view:
        vertices, faces = mesh.vertices, mesh.faces
        figure = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                   i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color="#b8b8b8",
                                   flatshading=False, lighting=dict(ambient=.4, diffuse=.8)))
        figure.update_layout(height=540, margin=dict(l=0, r=0, t=0, b=0),
                             scene=dict(aspectmode="data", xaxis_title="X · mm", yaxis_title="추정 깊이 · mm", zaxis_title="높이 · mm"))
        st.plotly_chart(figure, width="stretch")
        st.image(generated.astype(np.uint8) * 255, caption="입체화에 사용한 형상", width=160)
        st.caption(f"닫힌 표면: {geometry['watertight']} · 분리된 입체: {geometry['components_3d']} · 입력 윤곽 유지 IoU: {geometry['projection_iou']:.3f}")
        if geometry["components_3d"] > 1:
            st.warning("분리된 입체가 포함돼 있어. 제작 전 연결 여부를 확인해줘.")
        with st.expander("형상 생성 기록과 입체 측정"):
            st.json(dict(recipe=recipe, geometry=geometry))
    st.subheader("5 · 모델과 실험 기록 저장")
    cols = st.columns(4)
    for column, ext in zip(cols[:3], ["stl", "obj", "glb"]):
        column.download_button(ext.upper(), mesh_bytes(mesh, ext), f"modern_cliche.{ext}",
                               mime={"stl": "application/octet-stream", "obj": "text/plain", "glb": "model/gltf-binary"}[ext])
    cols[3].download_button("실험 전체 ZIP", experiment_zip(used, analysis, st.session_state.query, mesh, recipe, geometry, generated),
                            "experiment.zip", "application/zip", type="primary")
except Exception as exc:
    st.error(f"이 조건의 입체를 만들지 못했어: {exc}")
    st.download_button("분석 기록 ZIP 저장", experiment_zip(used, analysis, st.session_state.query), "experiment.zip", "application/zip")
