from __future__ import annotations
from io import BytesIO
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from skimage.feature import hog
from skimage.measure import marching_cubes
import trimesh

st.set_page_config(page_title='EIGENDOOR', layout='wide')
st.title('EIGENDOOR')
st.caption('DOOR-SCENE DATASET → SORTING → EIGENDOOR → 3D PHYSICALIZATION')
st.write('실제 이미지들을 동일한 조건으로 정규화한 뒤, PCA 및 유사도 분석을 통해 공통성과 불일치를 2D·3D 형태로 시각화하는 연구용 프로토타입입니다. 본 프로토타입은 생성형 이미지 AI를 사용하지 않으며, 실제 수집 이미지 또는 사용자가 업로드한 이미지를 기반으로 작동합니다.')
st.info('현재 방향은 비상구 표지 하나에 한정되지 않습니다. 문에 부착된 각종 사인, 경고문, 방번호, 접근 통제 장치, 손잡이, 인터폰, 도어 클로저, 각종 패치와 흔적을 함께 수집하여, “21세기의 지옥의 문”에 가까운 집합적 문 장면을 분석하는 것을 목표로 합니다. 가능하면 문짝, 문틀, 벽, 사인, 부착물, 주변 설비가 함께 보이는 이미지를 사용하십시오.')

WM_API = 'https://commons.wikimedia.org/w/api.php'

def search_wikimedia(q, n=24):
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': q,
        'gsrnamespace': 6,
        'gsrlimit': min(int(n), 50),
        'prop': 'imageinfo',
        'iiprop': 'url',
        'iiurlwidth': 512,
        'format': 'json',
        'origin': '*'
    }
    r = requests.get(WM_API, params=params, timeout=20, headers={'User-Agent': 'EIGENDOOR/0.4'})
    r.raise_for_status()
    out = []
    pages = r.json().get('query', {}).get('pages', {})
    for _, x in pages.items():
        ii = (x.get('imageinfo') or [{}])[0]
        thumb = ii.get('thumburl') or ii.get('url')
        url = ii.get('descriptionurl') or ii.get('url') or ''
        if thumb:
            out.append({
                'title': x.get('title') or 'Untitled',
                'thumb': thumb,
                'url': url,
                'license': 'Wikimedia Commons',
                'source': 'wikimedia'
            })
    return out

def download_items(items):
    ims = []; meta = []
    for x in items:
        try:
            r = requests.get(x['thumb'], timeout=15, headers={'User-Agent': 'EIGENDOOR/0.4'})
            r.raise_for_status()
            ims.append(Image.open(BytesIO(r.content)).convert('RGB'))
            meta.append(x)
        except Exception:
            pass
    return ims, meta

def demo_door_scene(i, size=160):
    rng = np.random.default_rng(i)
    im = Image.new('RGB', (size, size), 'white')
    d = ImageDraw.Draw(im)

    wall = int(226 + rng.uniform(-18, 12))
    d.rectangle([0, 0, size, size], fill=(wall, wall, wall))

    door_w = int(size * (0.34 + rng.uniform(-0.05, 0.07)))
    door_h = int(size * (0.62 + rng.uniform(-0.05, 0.05)))
    door_x = int(size * (0.50 + rng.uniform(-0.10, 0.08)))
    door_y = int(size * (0.22 + rng.uniform(-0.01, 0.06)))
    door_color = int(176 + rng.uniform(-35, 25))
    d.rectangle([door_x, door_y, door_x + door_w, door_y + door_h], fill=(door_color, door_color, door_color), outline='black', width=3)
    d.rectangle([door_x - 3, door_y - 3, door_x + door_w + 3, door_y + door_h + 3], outline='black', width=2)

    hx = door_x + door_w - int(door_w * 0.18)
    hy = door_y + int(door_h * (0.45 + rng.uniform(-0.08, 0.08)))
    d.rectangle([hx, hy, hx + 4, hy + 16], fill='black')

    if i % 2 == 0:
        d.rectangle([door_x + 6, door_y + 4, door_x + door_w - 6, door_y + 10], fill='black')

    num_signs = 1 + (i % 4)
    for k in range(num_signs):
        mode = (i + k) % 5
        if mode == 0:
            sx = door_x - 22 if k % 2 == 0 else door_x + door_w + 6
            sy = door_y + 20 + 22 * k
            d.rectangle([sx, sy, sx + 18, sy + 12], fill=(245, 245, 245), outline='black', width=2)
            d.line([(sx + 4, sy + 6), (sx + 14, sy + 6)], fill='black', width=1)
        elif mode == 1:
            sx = door_x + int(door_w * 0.20)
            sy = door_y - 16 - 12 * k
            d.rectangle([sx, sy, sx + 30, sy + 12], fill=(245, 245, 245), outline='black', width=2)
            d.line([(sx + 4, sy + 6), (sx + 22, sy + 6)], fill='black', width=1)
        elif mode == 2:
            sx = door_x - 18
            sy = door_y + int(door_h * 0.40)
            d.rectangle([sx, sy, sx + 12, sy + 20], fill=(245, 245, 245), outline='black', width=2)
            for yy in range(sy + 4, sy + 17, 4):
                d.line([(sx + 3, yy), (sx + 9, yy)], fill='black', width=1)
        elif mode == 3:
            sx = door_x + int(door_w * 0.30)
            sy = door_y - 14
            d.rectangle([sx, sy, sx + 26, sy + 14], fill=(245, 245, 245), outline='black', width=2)
            cx = sx + 9; cy = sy + 8
            d.ellipse([cx - 2, cy - 5, cx + 2, cy - 1], fill='black')
            d.line([(cx, cy - 1), (cx, cy + 4)], fill='black', width=1)
            d.line([(cx, cy + 4), (cx + 5, cy + 7)], fill='black', width=1)
            d.line([(sx + 14, sy + 8), (sx + 21, sy + 8)], fill='black', width=1)
            d.polygon([(sx + 21, sy + 8), (sx + 17, sy + 5), (sx + 17, sy + 11)], fill='black')
        else:
            sx = door_x + 6 + 10 * k
            sy = door_y + int(door_h * 0.18) + 16 * k
            d.rectangle([sx, sy, sx + 16, sy + 10], fill=(245, 245, 245), outline='black', width=1)

    if i % 3 == 0:
        d.rectangle([12, 44, 30, 82], fill=(242, 242, 242), outline='black', width=2)
    elif i % 3 == 1:
        d.ellipse([size - 28, 44, size - 12, 60], outline='black', width=2)
    else:
        d.line([(0, door_y + door_h + 8), (size, door_y + door_h + 8)], fill='black', width=2)

    return im

def normalize(im, size=96):
    pil = ImageOps.autocontrast(im.convert('L'))
    arr = np.asarray(pil, dtype=np.float32) / 255
    try:
        t = float(threshold_otsu(arr))
    except Exception:
        t = .5
    dark = arr < t
    light = arr >= t
    def border_fill(m):
        return np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
    mask = dark if border_fill(dark) <= border_fill(light) else light
    if mask.any():
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        pil = pil.crop((max(0, x0 - 4), max(0, y0 - 4), min(pil.width, x1 + 4), min(pil.height, y1 + 4)))
    pil.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (size, size), 255)
    canvas.paste(pil, ((size - pil.width) // 2, (size - pil.height) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255
    try:
        t = float(threshold_otsu(a))
    except Exception:
        t = .5
    m = a < t
    return m.astype(np.float32)

def analyze(images, size=96):
    data = np.stack([normalize(im, size) for im in images])
    feats = np.array([
        hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        for x in data
    ])
    X = StandardScaler().fit_transform(feats)
    k = max(2, min(8, len(data) - 1, X.shape[1]))
    emb = PCA(k, random_state=0).fit_transform(X)
    flat = data.reshape(len(data), -1)
    kp = max(1, min(6, len(data) - 1, flat.shape[1]))
    p = PCA(kp, random_state=0).fit(flat)
    order = np.argsort(emb[:, 0])
    mean = flat.mean(0).reshape(size, size)
    eig = p.components_.reshape(kp, size, size)
    return data, emb, order, mean, eig

def entropy_map(data):
    p = np.clip(data.mean(0), 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def img_from_arr(a, invert=False):
    a = np.asarray(a, dtype=float)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    if not invert:
        a = 1 - a
    return Image.fromarray(np.uint8(np.clip(a * 255, 0, 255)))

def mesh_from_field(field, depth=48):
    f = np.asarray(field, float)
    f = (f - f.min()) / (f.max() - f.min() + 1e-9)
    vol = np.zeros((depth, *f.shape), np.float32)
    for z in range(depth):
        th = abs((z / max(1, depth - 1)) * 2 - 1)
        vol[z] = (f >= th).astype(np.float32)
    v, fa, _, _ = marching_cubes(np.pad(vol, 1), level=.5)
    m = trimesh.Trimesh(v, fa, process=True)
    m.apply_translation(-m.bounding_box.centroid)
    return m

def mesh_fig(mesh):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], flatshading=True))
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=20, b=0), height=560)
    return fig

for k, v in {'images': [], 'meta': [], 'analysis': None, 'mesh': None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.header('1. DATASET')
mode = st.radio('데이터 소스', ['Wikimedia Commons search', '이미지 업로드', '데모 문 장면 데이터셋'], horizontal=True)
if mode == 'Wikimedia Commons search':
    q = st.text_input('검색어', 'door sign notice plaque warning sticker intercom access control')
    n = st.slider('최대 결과 수', 8, 60, 24, 4)
    if st.button('SEARCH', type='primary'):
        with st.spinner('Wikimedia Commons에서 이미지 검색 중…'):
            try:
                items = search_wikimedia(q, n)
                ims, meta = download_items(items)
            except Exception as e:
                st.error(str(e))
                ims, meta = [], []
        st.session_state.images = ims
        st.session_state.meta = meta
        st.session_state.analysis = None
        st.session_state.mesh = None
elif mode == '이미지 업로드':
    files = st.file_uploader('PNG / JPG / WEBP', type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
    if files:
        ims = []
        meta = []
        for f in files:
            try:
                ims.append(Image.open(f).convert('RGB'))
                meta.append({'title': f.name, 'source': 'upload', 'license': 'user-provided', 'url': ''})
            except Exception:
                pass
        st.session_state.images = ims
        st.session_state.meta = meta
        st.session_state.analysis = None
        st.session_state.mesh = None
else:
    if st.button('LOAD DEMO', type='primary'):
        st.session_state.images = [demo_door_scene(i) for i in range(36)]
        st.session_state.meta = [{'title': f'demo_door_scene_{i:03d}', 'source': 'synthetic demo scene', 'license': 'prototype', 'url': ''} for i in range(36)]
        st.session_state.analysis = None
        st.session_state.mesh = None

ims = st.session_state.images
if ims:
    st.write(f'**{len(ims)} images loaded**')
    st.caption('권장: 문, 문틀, 벽, 사인, 경고문, 방번호, 인터폰, 접근 통제 장치, 손잡이 등 문 주변 요소가 함께 드러나는 장면 이미지를 사용하십시오.')
    cols = st.columns(8)
    for i, im in enumerate(ims[:32]):
        cols[i % 8].image(im, use_container_width=True)
else:
    st.info('이미지를 검색하거나 업로드해주십시오.')

st.header('2. ANALYSIS')
size = st.select_slider('정규화 해상도', [64, 96, 128], 96)
if st.button('데이터셋 분석', type='primary', disabled=len(ims) < 3):
    with st.spinner('정규화 → HOG 특징 추출 → PCA 분석 수행 중…'):
        st.session_state.analysis = analyze(ims, size)
        st.session_state.mesh = None

if st.session_state.analysis is not None:
    data, emb, order, mean, eig = st.session_state.analysis
    c1, c2, c3 = st.columns(3)
    c1.metric('샘플 수', len(data))
    c2.metric('특징 공간', 'HOG + PCA')
    c3.metric('PCA 차원 수', emb.shape[1])

    fig = go.Figure(go.Scatter3d(
        x=emb[:, 0],
        y=emb[:, 1] if emb.shape[1] > 1 else np.zeros(len(emb)),
        z=emb[:, 2] if emb.shape[1] > 2 else np.zeros(len(emb)),
        mode='markers',
        text=[m.get('title', '') for m in st.session_state.meta]
    ))
    fig.update_layout(scene=dict(aspectmode='data'), height=480, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    a.subheader('MEAN DOOR SCENE')
    a.image(img_from_arr(mean), use_container_width=True)
    b.subheader('DISAGREEMENT / ENTROPY MAP')
    ent = entropy_map(data)
    b.image(img_from_arr(ent, invert=True), use_container_width=True)

    st.subheader('EIGENDOOR COMPONENTS')
    cols = st.columns(len(eig))
    for i, e in enumerate(eig):
        cols[i].image(img_from_arr(e), caption=f'PC {i + 1}', use_container_width=True)

    st.subheader('SORTED DOOR SCENES')
    cols = st.columns(9)
    for j, idx in enumerate(order[:36]):
        cols[j % 9].image(img_from_arr(data[idx]), use_container_width=True)

    df = pd.DataFrame([
        {
            **m,
            'PC1': float(emb[i, 0]),
            'PC2': float(emb[i, 1]) if emb.shape[1] > 1 else 0.0
        }
        for i, m in enumerate(st.session_state.meta)
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button('CSV 다운로드', df.to_csv(index=False).encode('utf-8-sig'), 'eigendoor_dataset.csv', 'text/csv')

    st.header('3. 3D PHYSICALIZATION')
    mapping = st.selectbox('3D 매핑 방식', ['UNCERTAINTY MASS', 'UNIVERSAL MASS'])
    depth = st.slider('깊이(Depth)', 24, 96, 48, 8)
    if st.button('3D 생성', type='primary'):
        field = entropy_map(data) if mapping == 'UNCERTAINTY MASS' else data.mean(0)
        with st.spinner('스칼라 필드를 메쉬로 변환하는 중…'):
            st.session_state.mesh = mesh_from_field(field, depth)
    if st.session_state.mesh is not None:
        mesh = st.session_state.mesh
        st.plotly_chart(mesh_fig(mesh), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric('정점 수', f'{len(mesh.vertices):,}')
        c2.metric('면 수', f'{len(mesh.faces):,}')
        st.download_button('STL 다운로드', mesh.export(file_type='stl'), 'eigendoor.stl', 'model/stl')

st.header('4. RESEARCH LOGIC')
st.markdown('''
**Research Question** — Are contemporary door surfaces becoming dense carriers of warnings, permissions, labels, controls, and traces, such that the modern door functions as a new information-laden threshold — a kind of 21st-century “Gate of Hell”?  

`real image dataset → normalization → feature extraction → PCA / sorting → mean door scene / eigendoor / entropy → 3D scalar field → mesh`  

In this version, the analytical unit is not a single sign but the **entire door scene** — including the door leaf, frame, wall, plaques, warning signs, access-control devices, hardware, and surrounding marks. The final grotesque form is therefore treated as an **outcome of disagreement and accumulation inside the dataset**, not as a style arbitrarily drawn in advance.
''')
