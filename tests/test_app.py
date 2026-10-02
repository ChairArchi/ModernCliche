from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_review_selection_persists_between_pages_and_modes_run():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py", default_timeout=90).run()
    assert not app.exception
    next(button for button in app.button if button.label == "검증용 16장 불러오기").click().run()
    next(button for button in app.button if button.label == "대상 영역 추출").click().run()
    app.checkbox[0].set_value(False).run()
    next(select for select in app.selectbox if select.label == "이미지 검토 페이지").set_value(1).run()
    next(select for select in app.selectbox if select.label == "이미지 검토 페이지").set_value(0).run()
    assert app.checkbox[0].value is False
    next(button for button in app.button if button.label == "선택한 데이터 분석").click().run()
    for mode in ("observed", "variation", "interpolate"):
        next(select for select in app.selectbox if select.label == "형상 원천").set_value(mode).run()
        assert not app.exception
        assert not app.error


def test_embedded_depth_analysis_and_surface_controls(monkeypatch):
    import numpy as np
    import modern_cliche.depth as module
    from modern_cliche.depth import backproject, depth_features
    def fake_inference(prepared, *args, **kwargs):
        y, x = np.mgrid[:48, :64]
        z = (2 + .1 * np.sin(x / 6) * np.cos(y / 5)).astype(np.float32)
        k = np.array([[.8, 0, .5], [0, 1.1, .5], [0, 0, 1]], dtype=np.float32)
        valid = np.zeros_like(z, bool);valid[5:-5,5:-5] = True
        result = dict(depth=z, points=backproject(z,k), intrinsics=k, valid=valid, foreground=valid,
            rgb=np.full((*z.shape,3),140,np.uint8), metadata=dict(model_id='fixture',revision='1',scale='synthetic',camera='fixture'))
        result['features'] = depth_features(result)
        return [result] * len(prepared['samples'])
    monkeypatch.setattr(module, 'infer_cohort', fake_inference)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'streamlit_app.py', default_timeout=90).run()
    next(b for b in app.button if b.label == '검증용 16장 불러오기').click().run()
    next(b for b in app.button if b.label == '대상 영역 추출').click().run()
    next(c for c in app.checkbox if c.label == '깊이 특징을 분류와 규칙 분석에 포함').set_value(True).run()
    next(b for b in app.button if b.label == '선택한 데이터 분석').click().run()
    assert not app.error and not app.exception
    assert app.session_state['analysis']['backend'] == 'shape+depth'
    next(r for r in app.radio if r.label == '3D 구성 방법').set_value('depth').run()
    assert not app.error and not app.exception
    next(n for n in app.number_input if n.label == '추가 두께 · mm · 0은 열린 표면').set_value(2.).run()
    assert not app.error and not app.exception
    next(s for s in app.selectbox if s.label == '표면 표시').set_value('source-color').run()
    assert not app.error and not app.exception
