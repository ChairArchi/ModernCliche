# Modern Cliché — EIGENSIGN

A research prototype for turning collections of supposedly universal pictograms into measurable visual data and 3D form.

Pipeline:

`keyword / images → normalization → PCA / similarity sorting → mean sign / eigensigns / entropy → 3D physicalization → STL`

The prototype does **not** use generative image AI. It works from real/openly licensed or user-provided images and deterministic data-processing rules.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app is designed for Streamlit Community Cloud deployment.
