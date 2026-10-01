# MODERN CLICHÉ · Shape Research 2.0

**대상 검색 → 이미지 수집·검토 → Sorting → 형상 규칙 탐색 → 3D 모델화**

같은 이름으로 모은 이미지의 형상 차이를 계산하고, 그 변형을 입체로 연결하는 Python 연구 도구야. 가고일·개·고양이 등 원하는 대상을 사용할 수 있어. 특정 디자인이나 괴물의 외형은 미리 정해두지 않았어.

## 실행

Python **3.11 또는 3.12**를 사용해. 첫 배경 분리 실행은 `u2netp` 모델 약 4.6MB를 내려받아.

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Streamlit Community Cloud: `ChairArchi/ModernCliche` → `main` → `streamlit_app.py`. Python 3.12를 권장해. 기존 앱도 같은 진입 파일을 사용해.

## 사용 흐름

1. 대상을 검색하거나 사진을 업로드해. Commons/Openverse를 선택할 수 있고, 검색 오류는 표시돼.
2. 대상 영역을 추출해. 주황색 영역을 검토하고 잘못된 이미지를 제외해. 수정 마스크 PNG도 사용할 수 있어.
3. 선택한 데이터를 분석해. 형상 그룹, 대표 이미지, 측정 특징, 그룹 차이, 분류 안정성, PCA 변형 방향을 비교해.
4. 관측 형상 / 그룹 안의 통계 변형 / 그룹 사이 보간 중 하나로 입체를 만들어. 깊이와 제작 크기는 별도 가정으로 기록돼.
5. STL·OBJ·GLB 또는 실험 전체 ZIP을 저장해. ZIP에는 원본·마스크·분석·출처·생성 규칙이 함께 들어 있어.

20장 이상을 권장해. 최소 3장으로도 탐색할 수 있지만 작은 표본에서 규칙을 확정하면 안 돼. 분류가 불안정하면 단일 그룹으로 유지해.

캡차처럼 이미 분류된 이미지에는 `filename,label` CSV를 함께 올릴 수 있어. 원래 라벨과 형상 그룹은 별도로 비교해.

## 기본 분석과 선택적 AI 특징

기본 실행은 **형태 통계 + signed-distance field + K-means + PCA**이고 CPU에서 작동해. 자동 배경 분리에는 사전 학습된 U²-Net을 사용해.

DINOv2/CLIP 분류 비교는 선택 사항이야. 설치하면 앱에 선택지가 나타나고 첫 실행은 모델 가중치를 내려받아. 메모리 여유가 있는 로컬 환경을 권장해.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ml.txt
```

## 명령줄과 재현

```bash
python -m modern_cliche.cli --query gargoyle --limit 24 --output output/gargoyle
python -m modern_cliche.cli --input ./images --output output/local
python -m modern_cliche.cli --experiment experiment.zip --output output/reproduced
```

실험 ZIP 복원은 저장된 마스크·특징을 사용해. 같은 패키지 버전에서 비교하면 좋아. CLI는 전체 실행을 재현하고 화면 복원은 데이터를 불러와 다시 분석할 수 있게 해.

## 방법의 범위

이 프로그램의 ‘규칙’은 수집한 실루엣 표본에서 측정한 공통점과 변형이야. **3D 깊이는 내부 반경을 확장하는 명시적 가정**이고 실제 대상의 입체 복원은 아니야. 생성 모델 학습·GAN 모드 붕괴·인간의 언캐니 반응을 측정했다고 주장하지 않아.

[방법론과 수식](docs/METHODOLOGY.md) · [논문과 구현의 대응](docs/REFERENCES.md)

[검증 결과와 실제 데이터 실행 기록](docs/VALIDATION.md)

## 검증

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

앱의 검증용 타원·십자는 계산 확인용 합성 데이터야. 실제 대상의 아카이브나 디자인 제안과 구분돼.
