# Dataset Collector

A separate personal-research utility for building large image corpora before running the Modern Cliché morphology pipeline.

## Workflow

1. Enter one or many multilingual search queries.
2. Search multiple sources:
   - DDGS image metasearch (Bing / DuckDuckGo)
   - Wikimedia Commons
   - Openverse
3. Merge candidate metadata and remove duplicate image URLs.
4. Filter by reported dimensions before downloading.
5. Review candidates visually in the browser, page by page.
6. Mark each image as Keep / Reject and assign a class.
7. Download only kept originals.
8. Re-check real image dimensions after download.
9. Remove exact duplicates (SHA-256) and near duplicates (difference hash).
10. Preserve source page, direct image URL, query, creator/license metadata when available.
11. Export a reproducible dataset folder and ZIP containing `images/`, `metadata.csv`, `metadata.json`, and `summary.json`.

Current manual classes:

- Torii
- Hongsalmun
- Paifang / Pailou
- Temple / Shrine Gate
- Monumental / Triumphal
- Stone / Post-and-Lintel
- Other

The initial class is inferred from the search query/title when possible and can be changed during review.

## Deploy as a web app

Streamlit Community Cloud can deploy this collector as a separate app from the same repository.

Use these GitHub coordinates:

```text
Repository: ChairArchi/ModernCliche
Branch: dataset-collector-v1
Entrypoint: tools/dataset_collector/app.py
```

The collector has its own lightweight dependency file at:

```text
tools/dataset_collector/requirements.txt
```

The web app is intended for:

```text
SEARCH -> REVIEW + CLASSIFY -> BUILD DATASET -> DOWNLOAD ZIP
```

Treat the cloud filesystem as temporary working storage. Download the finished ZIP to local storage after each collection session.

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

Use the collector to gather a broad candidate pool first, curate it visually, and then download only the selected originals. The main morphology tool should receive the resulting ZIP rather than performing web collection itself.

The collector intentionally does **not** circumvent logins, paywalls, anti-bot systems, access denials, or other technical restrictions. Source provenance is kept in the metadata so material can be reviewed or replaced later.
