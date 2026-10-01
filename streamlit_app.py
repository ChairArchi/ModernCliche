from __future__ import annotations
from io import BytesIO
import math
import random
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from skimage.filters import threshold_otsu
from skimage.feature import hog
from skimage.measure import marching_cubes
import trimesh

st.set_page_config(page_title='Modern Cliché — Uncanny Chimera Generator', layout='wide')
st.title('MODERN CLICHÉ')
st.caption('KEYWORD DATASET → SORTING / CLUSTERING → MODE COLLAPSE → UNCANNY CHIMERA → 3D PHYSICALIZATION')
st.write('실제 이미지들을 동일한 조건으로 정규화한 뒤, 특징 추출·정렬·군집화 과정을 거쳐, 생성 모델의 “모드 붕괴(mode collapse)”를 조형 원리로 전환하는 연구용 프로토타입입니다. 본 프로토타입은 생성형 이미지 AI를 사용하지 않으며, 실제 수집 이미지 또는 사용자가 업로드한 이미지를 기반으로 작동합니다.')
st.info('핵심 아이디어는 “키워드로 묶인 이미지 데이터셋이 여러 시각적 mode를 가진다”는 점입니다. 이 앱은 그 다양성을 일부러 붕괴시켜, 익숙한 범주가 익숙하지 않은 방식으로 수렴하는 언캐니한 키메라를 생성합니다.')

WM_API='https://commons.wikimedia.org/w/api.php'

def search_wikimedia_single(query, limit=10):
    params={'action':'query','generator':'search','gsrsearch':query,'gsrnamespace':6,'gsrlimit':min(int(limit),50),'prop':'imageinfo','iiprop':'url','iiurlwidth':512,'format':'json','origin':'*'}
    r=requests.get(WM_API,params=params,timeout=20,headers={'User-Agent':'ModernCliche/0.6'})
    r.raise_for_status()
    out=[]
    pages=r.json().get('query',{}).get('pages',{})
    for _,x in pages.items():
        ii=(x.get('imageinfo') or [{}])[0]
        thumb=ii.get('thumburl') or ii.get('url')
        url=ii.get('descriptionurl') or ii.get('url') or ''
        if thumb:
            out.append({'title':x.get('title') or 'Untitled','thumb':thumb,'url':url,'license':'Wikimedia Commons','source':'wikimedia','query':query})
    return out

def search_wikimedia_multi(query,use_pack=True,total_limit=30):
    base=query.strip()
    if not base:return []
    queries=[base]
    if use_pack:
        ql=base.lower()
        if 'exit' in ql or 'sign' in ql: queries += ['exit sign','emergency exit sign','wayfinding pictogram','safety sign']
        elif 'toilet' in ql or 'restroom' in ql: queries += ['toilet sign','restroom pictogram','washroom sign']
        elif 'door' in ql: queries += ['door sign','warning sign','access control panel','intercom']
        elif 'traffic' in ql: queries += ['traffic sign','pedestrian sign','road sign']
        else: queries += [f'{base} sign',f'{base} icon',f'{base} symbol']
    queries=list(dict.fromkeys(queries)); per=max(4,math.ceil(total_limit/max(1,len(queries))))
    merged=[]; seen=set()
    for q in queries:
        try: items=search_wikimedia_single(q,per)
        except Exception: items=[]
        for it in items:
            key=it['thumb']
            if key not in seen:
                seen.add(key); merged.append(it)
    if len(merged)<max(8,total_limit//4) and base.lower()!='door':
        try:
            more=search_wikimedia_single(base.split()[0],total_limit)
            for it in more:
                key=it['thumb']
                if key not in seen:
                    seen.add(key); merged.append(it)
        except Exception: pass
    return merged[:total_limit]

def download_items(items):
    ims=[]; meta=[]
    for x in items:
        try:
            r=requests.get(x['thumb'],timeout=15,headers={'User-Agent':'ModernCliche/0.6'}); r.raise_for_status()
            ims.append(Image.open(BytesIO(r.content)).convert('RGB')); meta.append(x)
        except Exception: pass
    return ims,meta

def demo_symbol(i,size=160):
    rng=np.random.default_rng(i); im=Image.new('RGB',(size,size),'white'); d=ImageDraw.Draw(im); kind=i%6; pad=18
    if kind in [0,1]:
        d.rectangle([pad,pad,size-pad,size-pad],outline='black',width=4); cx,cy=size*.42,size*.48
        d.ellipse([cx-7,cy-26,cx+7,cy-12],fill='black'); d.line([(cx,cy-12),(cx,cy+6)],fill='black',width=4)
        d.line([(cx,cy-4),(cx+20,cy+2)],fill='black',width=4); d.line([(cx,cy-4),(cx-14,cy+2)],fill='black',width=4)
        d.line([(cx,cy+6),(cx+16,cy+26)],fill='black',width=4); d.line([(cx,cy+6),(cx-12,cy+24)],fill='black',width=4)
        if i%2==0:
            d.line([(size*.58,size*.52),(size*.82,size*.52)],fill='black',width=5); d.polygon([(size*.82,size*.52),(size*.72,size*.46),(size*.72,size*.58)],fill='black')
        else:
            d.line([(size*.18,size*.52),(size*.42,size*.52)],fill='black',width=5); d.polygon([(size*.18,size*.52),(size*.28,size*.46),(size*.28,size*.58)],fill='black')
    elif kind==2:
        d.rectangle([pad,pad,size-pad,size-pad],outline='black',width=3)
        for x in [size*.34,size*.66]:
            d.ellipse([x-8,size*.22-8,x+8,size*.22+8],fill='black'); d.line([(x,size*.30),(x,size*.58)],fill='black',width=5)
            d.line([(x-16,size*.40),(x+16,size*.40)],fill='black',width=5); d.line([(x,size*.58),(x-12,size*.82)],fill='black',width=5); d.line([(x,size*.58),(x+12,size*.82)],fill='black',width=5)
    elif kind==3:
        tri=[(size*.5,pad),(size-pad,size-pad),(pad,size-pad)]; d.polygon(tri,outline='black',fill=None,width=4)
        d.rectangle([size*.30,size*.40,size*.68,size*.52],fill='black'); d.line([(size*.50,size*.40),(size*.60,size*.28)],fill='black',width=5)
    elif kind==4:
        d.rectangle([pad,pad,size-pad,size-pad],outline='black',width=3); d.rectangle([size*.40,size*.30,size*.62,size*.76],outline='black',width=5)
        d.line([(size*.52,size*.30),(size*.62,size*.22)],fill='black',width=4); d.arc([size*.30,size*.18,size*.58,size*.44],300,60,fill='black',width=4)
    else:
        pts=[]
        for a in np.linspace(0,2*np.pi,10,endpoint=False):
            r=size*.28+rng.uniform(-8,10); pts.append((size/2+r*np.cos(a),size/2+r*np.sin(a)))
        d.polygon(pts,outline='black'); d.ellipse([size*.38,size*.42,size*.45,size*.50],outline='black',width=3); d.ellipse([size*.55,size*.42,size*.62,size*.50],outline='black',width=3); d.arc([size*.42,size*.56,size*.58,size*.68],0,180,fill='black',width=3)
    for _ in range(i%4):
        x=rng.integers(8,size-22); y=rng.integers(8,size-16); d.rectangle([x,y,x+12,y+6],outline='black',width=1)
    return im

def normalize(im,size=96):
    pil=ImageOps.autocontrast(im.convert('L')); arr=np.asarray(pil,dtype=np.float32)/255.0
    try:t=float(threshold_otsu(arr))
    except Exception:t=.5
    dark=arr<t; light=arr>=t
    def border_fill(m): return np.concatenate([m[0],m[-1],m[:,0],m[:,-1]]).mean()
    mask=dark if border_fill(dark)<=border_fill(light) else light
    if mask.any():
        ys,xs=np.where(mask); x0,x1=xs.min(),xs.max()+1; y0,y1=ys.min(),ys.max()+1
        pil=pil.crop((max(0,x0-4),max(0,y0-4),min(pil.width,x1+4),min(pil.height,y1+4)))
    pil.thumbnail((size,size),Image.Resampling.LANCZOS); canvas=Image.new('L',(size,size),255); canvas.paste(pil,((size-pil.width)//2,(size-pil.height)//2))
    a=np.asarray(canvas,dtype=np.float32)/255.0
    try:t2=float(threshold_otsu(a))
    except Exception:t2=.5
    return (a<t2).astype(np.float32)

def extract_features(mask):
    h=hog(mask,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)
    small=mask.reshape(mask.shape[0]//4,4,mask.shape[1]//4,4).mean(axis=(1,3)).ravel(); density=np.array([mask.mean()])
    ys,xs=np.where(mask>.5); centroid=np.array([xs.mean()/mask.shape[1],ys.mean()/mask.shape[0]]) if len(xs)>0 else np.array([.5,.5])
    return np.concatenate([h,small,density,centroid])

def analyze(images,size=96,n_clusters=4):
    data=np.stack([normalize(im,size) for im in images]); feats=np.stack([extract_features(x) for x in data]); X=StandardScaler().fit_transform(feats)
    k_pca=max(2,min(8,len(data)-1,X.shape[1])); emb=PCA(k_pca,random_state=0).fit_transform(X)
    k=max(2,min(n_clusters,len(data))); km=KMeans(n_clusters=k,random_state=0,n_init=10); labels=km.fit_predict(X); centers=km.cluster_centers_
    nearest,_=pairwise_distances_argmin_min(centers,X); cluster_means=np.stack([data[labels==i].mean(axis=0) for i in range(k)])
    freqs=np.array([(labels==i).mean() for i in range(k)]); order=np.argsort(emb[:,0]); global_mean=data.mean(axis=0)
    pcs=PCA(min(6,len(data),size*size),random_state=0).fit(data.reshape(len(data),-1)).components_.reshape(-1,size,size)
    return {'data':data,'feats':feats,'emb':emb,'labels':labels,'cluster_means':cluster_means,'cluster_freqs':freqs,'nearest_idx':nearest,'order':order,'global_mean':global_mean,'pcs':pcs}

def generate_chimera(cluster_means,freqs,target_idx=0,collapse=.75,strips=7,residual_boost=.30,seed=0):
    rng=random.Random(seed); k,h,w=cluster_means.shape; one_hot=np.zeros(k,dtype=float); one_hot[target_idx]=1.0
    weights=(1.0-collapse)*freqs+collapse*one_hot; weights=weights/weights.sum(); collapsed_field=np.tensordot(weights,cluster_means,axes=(0,0))
    chimera=np.zeros((h,w),dtype=float); x_edges=np.linspace(0,w,strips+1).astype(int); y_mid=h//2
    for s in range(strips):
        x0,x1=x_edges[s],x_edges[s+1]
        if rng.random()<collapse*.8: c=target_idx if rng.random()<.65 else rng.choices(range(k),weights=weights,k=1)[0]
        else: c=rng.choices(range(k),weights=weights,k=1)[0]
        region=cluster_means[c][:,x0:x1].copy()
        if k>1 and rng.random()<.45:
            c2=rng.choices(range(k),weights=weights,k=1)[0]; mix_y=int(y_mid+rng.randint(-h//6,h//6))
            region[:mix_y,:]=cluster_means[c][:mix_y,x0:x1]; region[mix_y:,:]=cluster_means[c2][mix_y:,x0:x1]
        chimera[:,x0:x1]=region
    residual=cluster_means[target_idx]-cluster_means.mean(axis=0)
    monster=(1-.55*collapse)*collapsed_field+(.55*collapse)*chimera+residual_boost*collapse*residual
    return np.clip(monster,0.0,1.0),collapsed_field,chimera,weights

def img_from_arr(a,invert=False):
    a=np.asarray(a,dtype=float); a=(a-a.min())/(a.max()-a.min()+1e-9)
    if not invert:a=1-a
    return Image.fromarray(np.uint8(np.clip(a*255,0,255)))

def mesh_from_field(field,depth=48):
    f=np.asarray(field,float); f=(f-f.min())/(f.max()-f.min()+1e-9); vol=np.zeros((depth,*f.shape),np.float32)
    for z in range(depth):
        th=abs((z/max(1,depth-1))*2-1); vol[z]=(f>=th).astype(np.float32)
    v,fa,_,_=marching_cubes(np.pad(vol,1),level=.5); m=trimesh.Trimesh(v,fa,process=True); m.apply_translation(-m.bounding_box.centroid); return m

def mesh_fig(mesh):
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces); fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],flatshading=True)); fig.update_layout(scene=dict(aspectmode='data'),margin=dict(l=0,r=0,t=20,b=0),height=540); return fig

for k,v in {'images':[],'meta':[],'analysis':None,'chimera':None,'mesh':None}.items():
    if k not in st.session_state: st.session_state[k]=v

st.header('1. DATASET')
mode=st.radio('데이터 소스',['Wikimedia Commons search','이미지 업로드','데모 심볼 데이터셋'],horizontal=True)
if mode=='Wikimedia Commons search':
    q=st.text_input('키워드','exit sign'); use_pack=st.checkbox('연관 검색어 팩 함께 사용',value=True); n=st.slider('최대 결과 수',8,60,24,4)
    if st.button('SEARCH',type='primary'):
        with st.spinner('이미지 검색 및 다운로드 중…'):
            try: items=search_wikimedia_multi(q,use_pack,n); ims,meta=download_items(items)
            except Exception as e: st.error(str(e)); ims,meta=[],[]
        st.session_state.images=ims; st.session_state.meta=meta; st.session_state.analysis=None; st.session_state.chimera=None; st.session_state.mesh=None
elif mode=='이미지 업로드':
    files=st.file_uploader('PNG / JPG / WEBP',type=['png','jpg','jpeg','webp'],accept_multiple_files=True)
    if files:
        ims=[]; meta=[]
        for f in files:
            try: ims.append(Image.open(f).convert('RGB')); meta.append({'title':f.name,'source':'upload','license':'user-provided','url':'','query':''})
            except Exception: pass
        st.session_state.images=ims; st.session_state.meta=meta; st.session_state.analysis=None; st.session_state.chimera=None; st.session_state.mesh=None
else:
    if st.button('LOAD DEMO',type='primary'):
        ims=[demo_symbol(i) for i in range(36)]; meta=[{'title':f'demo_symbol_{i:03d}','source':'synthetic demo','license':'prototype','url':'','query':'demo'} for i in range(36)]
        st.session_state.images=ims; st.session_state.meta=meta; st.session_state.analysis=None; st.session_state.chimera=None; st.session_state.mesh=None

ims=st.session_state.images
if ims:
    st.write(f'**{len(ims)} images loaded**'); st.caption('형태 차이는 있으나 시각적으로 같은 범주로 읽히는 이미지를 사용할수록 결과가 좋습니다. 예: exit sign, toilet sign, CCTV warning, fire extinguisher, mask, intercom 등.')
    cols=st.columns(8)
    for i,im in enumerate(ims[:32]): cols[i%8].image(im,use_container_width=True)
else: st.info('이미지를 검색하거나 업로드해주십시오.')

st.header('2. ANALYSIS')
col_a,col_b=st.columns([1,1]); size=col_a.select_slider('정규화 해상도',[64,96,128],value=96); n_clusters=col_b.slider('군집 수(Mode 수)',2,8,4,1)
if st.button('데이터셋 분석',type='primary',disabled=len(ims)<4):
    with st.spinner('정규화 → 특징 추출 → PCA / K-Means 수행 중…'): st.session_state.analysis=analyze(ims,size=size,n_clusters=n_clusters); st.session_state.chimera=None; st.session_state.mesh=None
A=st.session_state.analysis
if A is not None:
    data=A['data']; emb=A['emb']; labels=A['labels']; cluster_means=A['cluster_means']; freqs=A['cluster_freqs']; order=A['order']; pcs=A['pcs']
    c1,c2,c3=st.columns(3); c1.metric('샘플 수',len(data)); c2.metric('추정된 mode 수',len(cluster_means)); c3.metric('PCA 차원 수',emb.shape[1])
    colors=[int(x) for x in labels]; fig=go.Figure(go.Scatter3d(x=emb[:,0],y=emb[:,1] if emb.shape[1]>1 else np.zeros(len(emb)),z=emb[:,2] if emb.shape[1]>2 else np.zeros(len(emb)),mode='markers',marker=dict(size=5,color=colors,colorscale='Viridis'),text=[m.get('title','') for m in st.session_state.meta])); fig.update_layout(scene=dict(aspectmode='data'),height=460,margin=dict(l=0,r=0,t=20,b=0)); st.plotly_chart(fig,use_container_width=True)
    st.subheader('MODE PROTOTYPES'); cols=st.columns(len(cluster_means))
    for i,cm in enumerate(cluster_means): cols[i].image(img_from_arr(cm),caption=f'Mode {i+1} · {freqs[i]*100:.1f}%',use_container_width=True)
    st.subheader('대표 정렬 결과'); cols=st.columns(9)
    for j,idx in enumerate(order[:36]): cols[j%9].image(img_from_arr(data[idx]),use_container_width=True)
    st.subheader('Principal Components'); cols=st.columns(min(6,len(pcs)))
    for i,e in enumerate(pcs[:6]): cols[i].image(img_from_arr(e),caption=f'PC {i+1}',use_container_width=True)
    df=pd.DataFrame([{**m,'mode':int(labels[i]),'PC1':float(emb[i,0]),'PC2':float(emb[i,1]) if emb.shape[1]>1 else 0.0,'PC3':float(emb[i,2]) if emb.shape[1]>2 else 0.0} for i,m in enumerate(st.session_state.meta)])
    st.dataframe(df,use_container_width=True,hide_index=True); st.download_button('CSV 다운로드',df.to_csv(index=False).encode('utf-8-sig'),'modern_cliche_dataset.csv','text/csv')
    st.header('3. UNCANNY CHIMERA GENERATOR')
    left,right=st.columns([1,1]); target_mode_ui=left.selectbox('붕괴 기준 Mode',['자동(가장 빈도 높은 mode)']+[f'Mode {i+1}' for i in range(len(cluster_means))]); collapse=left.slider('Mode Collapse 강도',0.0,1.0,.72,.01); strips=right.slider('키메라 분절 수',3,15,7,1); residual_boost=right.slider('차이 증폭 강도',0.0,1.0,.30,.01); seed=right.number_input('랜덤 시드',value=0,step=1)
    target_idx=int(np.argmax(freqs)) if target_mode_ui.startswith('자동') else int(target_mode_ui.split()[-1])-1
    if st.button('키메라 생성',type='primary'):
        monster,collapsed_field,chimera_field,weights=generate_chimera(cluster_means,freqs,target_idx,collapse,strips,residual_boost,int(seed)); st.session_state.chimera={'monster':monster,'collapsed_field':collapsed_field,'chimera_field':chimera_field,'weights':weights,'target_idx':target_idx}; st.session_state.mesh=None
    C=st.session_state.chimera
    if C is not None:
        weight_df=pd.DataFrame({'mode':[f'Mode {i+1}' for i in range(len(freqs))],'original_freq':freqs,'collapsed_weight':C['weights']}); st.write('**Collapse weights** — 각 mode가 최종 형상에 얼마나 기여하는지 보여줍니다.'); st.dataframe(weight_df,use_container_width=True,hide_index=True)
        c1,c2,c3=st.columns(3); c1.subheader('Collapsed Mean'); c1.image(img_from_arr(C['collapsed_field']),use_container_width=True); c2.subheader('Chimera Composition'); c2.image(img_from_arr(C['chimera_field']),use_container_width=True); c3.subheader('Final Uncanny Monster'); c3.image(img_from_arr(C['monster']),use_container_width=True)
        buf=BytesIO(); img_from_arr(C['monster']).save(buf,format='PNG'); st.download_button('PNG 다운로드',buf.getvalue(),'uncanny_chimera.png','image/png')
        st.header('4. 3D PHYSICALIZATION'); depth=st.slider('깊이(Depth)',24,96,48,8)
        if st.button('3D 생성',type='primary'): st.session_state.mesh=mesh_from_field(C['monster'],depth=depth)
        if st.session_state.mesh is not None:
            mesh=st.session_state.mesh; st.plotly_chart(mesh_fig(mesh),use_container_width=True); m1,m2=st.columns(2); m1.metric('정점 수',f'{len(mesh.vertices):,}'); m2.metric('면 수',f'{len(mesh.faces):,}'); st.download_button('STL 다운로드',mesh.export(file_type='stl'),'uncanny_chimera.stl','model/stl')

st.header('5. RESEARCH LOGIC')
st.markdown('''**Research Question** — When a keyword gathers multiple visually related images, can the instability between their modes be converted into an uncanny chimera rather than a stable average?\n\n`keyword dataset → normalization → feature extraction → PCA / clustering → mode prototypes → intentional mode collapse → uncanny chimera → 3D mesh`\n\n본 프로토타입에서 괴물은 임의의 스타일이 아니라, **데이터셋 내부 다양성을 일부러 붕괴시키는 과정**에서 생성되는 결과입니다. 즉, 익숙한 범주가 낯선 방식으로 수렴할 때 발생하는 어긋남을 조형적 핵심으로 다룹니다.''')