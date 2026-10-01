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
