# Deploying the dashboard (free)

The dashboard reads committed files only (`results/`, and `runs/` if present), so it needs no GPU,
no secrets and no data download. Raw IP102 images are never in the repo or the dashboard.

## Streamlit Community Cloud

1. Sign in at share.streamlit.io with the GitHub account that owns the repo, and grant it access to
   `Nathaniel47/fedsurveil` (the repo is public).
2. New app, branch `main`, main file `app/dashboard.py`, Python 3.11 under Advanced settings.
3. It installs `requirements.txt` (streamlit, plotly, numpy, pandas: no torch). No secrets needed.
4. Check the "Final results (BloodMNIST)" tab shows three charts and the table.

## Hugging Face Spaces (alternative)

Create a Space with the Streamlit SDK, push this repo to it, and set `app_file: app/dashboard.py`
in the Space README header. It uses the same `requirements.txt`.

## Before sharing the link

- The dashboard says the secure aggregation is a simulation, and every privacy number names its
  guarantee. Keep that wording if you edit it.
- Only BloodMNIST results are shown; do not describe the project as covering IP102 yet.
- Free hosts put apps to sleep when idle, so the first visit may take a minute.
