# Dataset Collector

A separate personal-research utility for building large image corpora before running the Modern Cliché morphology pipeline.

## What it does

1. Accepts one or many multilingual search queries.
2. Searches multiple sources:
   - DDGS image metasearch (Bing / DuckDuckGo)
   - Wikimedia Commons
   - Openverse
3. Merges candidate metadata and removes duplicate image URLs.
4. Filters by reported dimensions before downloading.
5. Downloads original raster images without bypassing access controls.
6. Re-checks real image dimensions after download.
7. Removes exact duplicates (SHA-256) and near duplicates (difference hash).
8. Preserves source page, direct image URL, query, creator/license metadata when available.
9. Exports a reproducible dataset folder and ZIP containing `images/`, `metadata.csv`, `metadata.json`, and `summary.json`.

## Run locally

From the repository root:

```bash
python -m venv .venv-collector
# Windows
.venv-collector\\Scripts\\activate
# macOS / Linux
# source .venv-collector/bin/activate

pip install -r tools/dataset_collector/requirements.txt
streamlit run tools/dataset_collector/app.py
```

Outputs are written to `tools/dataset_collector/output/` by default. Override with `MC_COLLECTOR_OUTPUT`.

## Collection strategy

Use the collector to gather a broad candidate pool, then let the program remove broken, low-resolution, duplicate, and near-duplicate images. The main morphology tool should receive the resulting ZIP rather than performing web collection itself.

The collector intentionally does **not** circumvent logins, paywalls, anti-bot systems, access denials, or other technical restrictions. Source provenance is kept in the metadata so material can be reviewed or replaced later.
