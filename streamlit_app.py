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

st.set_page_config(page_title='EIGENSIGN', layout='wide')
st.title('EIGENSIGN')
st.caption('UNIVERSAL ICON → DATASET → SORTING → EIGENSIGN → 3D PHYSICALIZATION')
st.write('실제 이미지들을 같은 조건으로 정규화하고 PCA/유사도 분석을 거쳐 공통성과 불일치를 2D·3D 형태로 드러내는 프로토타입이야. 생성형 이미지 AI는 사용하지 않아.')
st.caption('Archive replay note: the original 2026 prototype used Openverse search. This replay falls back to Wikimedia Commons because the archived Openverse request now returns HTTP 401 on Streamlit Cloud.')

API='https://commons.wikimedia.org/w/api.php'

def search_openverse(q, n=24):
    params={
        'action':'query',
        'generator':'search',
        'gsrsearch':q,
        'gsrnamespace':6,
        'gsrlimit':min(int(n),50),
        'prop':'imageinfo',
        'iiprop':'url',
        'iiurlwidth':512,
        'format':'json',
        'origin':'*',
    }
    r=requests.get(API,params=params,timeout=20,headers={'User-Agent':'EIGENSIGN-Archive/0.2'})
    r.raise_for_status()
    out=[]
    pages=r.json().get('query',{}).get('pages',{})
    for x in pages.values():
        ii=(x.get('imageinfo') or [{}])[0]
        u=ii.get('thumburl') or ii.get('url')
        if u:
            out.append({
                'title':x.get('title') or 'Untitled',
                'thumb':u,
                'url':ii.get('descriptionurl') or ii.get('url') or '',
                'license':'Wikimedia Commons',
                'source':'wikimedia',
            })
    return out

def download_items(items):
    ims=[]; meta=[]
    for x in items:
        try:
            r=requests.get(x['thumb'],timeout=15,headers={'User-Agent':'EIGENSIGN/0.2'}); r.raise_for_status()
            ims.append(Image.open(BytesIO(r.content)).convert('RGB')); meta.append(x)
        except Exception: pass
    return ims,meta

def demo_sign(i,size=160):
    rng=np.random.default_rng(i); im=Image.new('RGB',(size,size),'white'); d=ImageDraw.Draw(im)
    door_x=int(size*(.64+rng.uniform(-.07,.07))); top=int(size*.2); bottom=int(size*.82)
    d.rectangle([door_x,top,door_x+int(size*.18),bottom],outline='black',width=5)
    cx=int(size*(.39+rng.uniform(-.07,.07))); cy=int(size*(.47+rng.uniform(-.04,.04))); lw=7; hr=8
    d.ellipse([cx-hr,cy-38-hr,cx+hr,cy-38+hr],fill='black'); sh=(cx,cy-18); hip=(cx,cy+14)
    d.line([sh,hip],fill='black',width=lw); d.line([sh,(cx+24,cy-4)],fill='black',width=lw); d.line([sh,(cx-18,cy-3)],fill='black',width=lw)
    d.line([hip,(cx+25,cy+38)],fill='black',width=lw); d.line([hip,(cx-19,cy+35)],fill='black',width=lw)
    ay=124
    if i%3==0:
        d.line([(20,ay),(65,ay)],fill='black',width=lw); d.polygon([(65,ay),(53,ay-10),(53,ay+10)],fill='black')
    elif i%3==1:
        d.line([(24,ay),(69,ay)],fill='black',width=lw); d.polygon([(24,ay),(36,ay-10),(36,ay+10)],fill='black')
    else:
        d.line([(39,138),(39,108)],fill='black',width=lw); d.polygon([(39,108),(29,120),(49,120)],fill='black')
    return im

def normalize(im,size=96):
    pil=ImageOps.autocontrast(im.convert('L')); arr=np.asarray(pil,dtype=np.float32)/255
    try:t=float(threshold_otsu(arr))
    except:t=.5
    dark=arr<t; light=arr>=t
    def bf(m): return np.concatenate([m[0],m[-1],m[:,0],m[:,-1]]).mean()
    mask=dark if bf(dark)<=bf(light) else light
    if mask.any():
        ys,xs=np.where(mask); x0,x1=xs.min(),xs.max()+1; y0,y1=ys.min(),ys.max()+1
        pil=pil.crop((max(0,x0-4),max(0,y0-4),min(pil.width,x1+4),min(pil.height,y1+4)))
    pil.thumbnail((size,size),Image.Resampling.LANCZOS); canvas=Image.new('L',(size,size),255); canvas.paste(pil,((size-pil.width)//2,(size-pil.height)//2))
    a=np.asarray(canvas,dtype=np.float32)/255
    try:t=float(threshold_otsu(a))
    except:t=.5
    m=a<t
    return m.astype(np.float32)

def analyze(images,size=96):
    data=np.stack([normalize(im,size) for im in images])
    feats=np.array([hog(x,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True) for x in data])
    X=StandardScaler().fit_transform(feats); k=max(2,min(8,len(data)-1,X.shape[1])); emb=PCA(k,random_state=0).fit_transform(X)
    flat=data.reshape(len(data),-1); kp=max(1,min(6,len(data)-1,flat.shape[1])); p=PCA(kp,random_state=0).fit(flat)
    order=np.argsort(emb[:,0]); mean=flat.mean(0).reshape(size,size); eig=p.components_.reshape(kp,size,size)
    return data,emb,order,mean,eig

def entropy_map(data):
    p=np.clip(data.mean(0),1e-6,1-1e-6); return -(p*np.log2(p)+(1-p)*np.log2(1-p))

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
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces)
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],flatshading=True))
    fig.update_layout(scene=dict(aspectmode='data'),margin=dict(l=0,r=0,t=20,b=0),height=560); return fig

for k,v in {'images':[],'meta':[],'analysis':None,'mesh':None}.items():
    if k not in st.session_state: st.session_state[k]=v

st.header('1. DATASET')
mode=st.radio('Source',['Openverse keyword search','Upload images','Demo EXIT dataset'],horizontal=True)
if mode=='Openverse keyword search':
    q=st.text_input('Keyword','emergency exit sign pictogram')
    n=st.slider('Max results',8,60,24,4)
    if st.button('SEARCH',type='primary'):
        with st.spinner('공개 라이선스 이미지 검색 중…'):
            try: items=search_openverse(q,n); ims,meta=download_items(items)
            except Exception as e: st.error(str(e)); ims=[]; meta=[]
        st.session_state.images=ims; st.session_state.meta=meta; st.session_state.analysis=None
elif mode=='Upload images':
    files=st.file_uploader('PNG / JPG / WEBP',type=['png','jpg','jpeg','webp'],accept_multiple_files=True)
    if files:
        ims=[]; meta=[]
        for f in files:
            try: ims.append(Image.open(f).convert('RGB')); meta.append({'title':f.name,'source':'upload','license':'user-provided','url':''})
            except: pass
        st.session_state.images=ims; st.session_state.meta=meta; st.session_state.analysis=None
else:
    if st.button('LOAD DEMO',type='primary'):
        st.session_state.images=[demo_sign(i) for i in range(36)]; st.session_state.meta=[{'title':f'demo_{i:03d}','source':'synthetic demo','license':'prototype','url':''} for i in range(36)]; st.session_state.analysis=None

ims=st.session_state.images
if ims:
    st.write(f'**{len(ims)} images loaded**'); cols=st.columns(8)
    for i,im in enumerate(ims[:32]): cols[i%8].image(im,use_container_width=True)
else: st.info('검색하거나 이미지를 업로드해줘.')

st.header('2. ANALYSIS')
size=st.select_slider('Normalize resolution',[64,96,128],96)
if st.button('ANALYZE DATASET',type='primary',disabled=len(ims)<3):
    with st.spinner('정규화 → HOG → PCA…'):
        st.session_state.analysis=analyze(ims,size)

if st.session_state.analysis is not None:
    data,emb,order,mean,eig=st.session_state.analysis
    c1,c2,c3=st.columns(3)
    c1.metric('Samples',len(data)); c2.metric('Feature space','HOG + PCA'); c3.metric('PCA dims',emb.shape[1])
    fig=go.Figure(go.Scatter3d(x=emb[:,0],y=emb[:,1] if emb.shape[1]>1 else 0,z=emb[:,2] if emb.shape[1]>2 else 0,mode='markers',text=[m.get('title','') for m in st.session_state.meta]))
    fig.update_layout(scene=dict(aspectmode='data'),height=480,margin=dict(l=0,r=0,t=20,b=0)); st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2); a.subheader('MEAN SIGN'); a.image(img_from_arr(mean),use_container_width=True); b.subheader('DISAGREEMENT / ENTROPY'); ent=entropy_map(data); b.image(img_from_arr(ent,invert=True),use_container_width=True)
    st.subheader('EIGENSIGNS')
    cols=st.columns(len(eig))
    for i,e in enumerate(eig): cols[i].image(img_from_arr(e),caption=f'PC {i+1}',use_container_width=True)
    st.subheader('SORTED SIGNS')
    cols=st.columns(9)
    for j,idx in enumerate(order[:36]): cols[j%9].image(img_from_arr(data[idx]),use_container_width=True)
    df=pd.DataFrame([{**m,'PC1':float(emb[i,0]),'PC2':float(emb[i,1]) if emb.shape[1]>1 else 0.0} for i,m in enumerate(st.session_state.meta)])
    st.dataframe(df,use_container_width=True,hide_index=True); st.download_button('DOWNLOAD CSV',df.to_csv(index=False).encode('utf-8-sig'),'eigensign_dataset.csv','text/csv')

    st.header('3. 3D PHYSICALIZATION')
    mapping=st.selectbox('Mapping',['UNCERTAINTY MASS','UNIVERSAL MASS'])
    depth=st.slider('Depth',24,96,48,8)
    if st.button('GENERATE 3D',type='primary'):
        field=entropy_map(data) if mapping=='UNCERTAINTY MASS' else data.mean(0)
        with st.spinner('scalar field → marching cubes mesh…'): st.session_state.mesh=mesh_from_field(field,depth)
    if st.session_state.mesh is not None:
        mesh=st.session_state.mesh; st.plotly_chart(mesh_fig(mesh),use_container_width=True)
        c1,c2=st.columns(2); c1.metric('Vertices',f'{len(mesh.vertices):,}'); c2.metric('Faces',f'{len(mesh.faces):,}')
        st.download_button('DOWNLOAD STL',mesh.export(file_type='stl'),'eigensign.stl','model/stl')

st.header('4. RESEARCH LOGIC')
st.markdown('''**Question** — Are supposedly universal icons actually one stable visual language, or a family of regional and historical variations that we merely recognize as equivalent?\n\n`real image dataset → normalization → feature extraction → PCA / sorting → mean / eigensign / entropy → 3D scalar field → mesh`\n\nThe grotesque form is treated as an **outcome of disagreement inside the dataset**, not as a style drawn in advance.''')
