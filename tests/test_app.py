from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_collect_screen_is_staged_and_english():
    app = AppTest.from_file(APP, default_timeout=90).run()
    assert not app.exception
    assert app.title[0].value == "MODERN CLICHÉ"
    assert any(field.label == "Subject" for field in app.text_input)
    assert any(button.label == "Collect images" for button in app.button)
    assert not any(button.label == "Discover groups" for button in app.button)
    assert not any(button.label == "Build statistical surface" for button in app.button)


def test_collect_screen_exposes_mixed_source_search():
    app = AppTest.from_file(APP, default_timeout=90).run()
    assert not app.exception
    source = next(select for select in app.selectbox if select.label == "Sources")
    assert source.value == "both"
    assert "Wikimedia Commons + Openverse" in source.options
    slider = next(slider for slider in app.select_slider if slider.label == "Candidate images")
    assert slider.value == 120
