# Deploying the dashboard (free)

The dashboard reads committed files only (`results/`, and `runs/` if present), so it needs no GPU,
no secrets and no data download. Raw IP102 images are never in the repo or the dashboard.

## Streamlit Community Cloud

Deployed on 2026-11-25 at https://fedsurveil-smwgy3kxghui5lqha38jve.streamlit.app (branch `main`, Python 3.11, no secrets).
The steps used:

1. Sign in at share.streamlit.io with the GitHub account that owns the repo, and grant it access to
   `Nathaniel47/fedsurveil` (the repo is public).
2. New app, branch `main`, main file `app/dashboard.py`, Python 3.11 under Advanced settings.
3. It installs `requirements.txt` (streamlit, plotly, numpy, pandas: no torch). No secrets needed.
4. Check the "Final results" tab shows the charts and the tables.

## Hugging Face Spaces (alternative)

Create a Space with the Streamlit SDK, push this repo to it, and set `app_file: app/dashboard.py`
in the Space README header. It uses the same `requirements.txt`.

## Before sharing the link

- The dashboard says the secure aggregation is a simulation, and every privacy number names its
  guarantee. Keep that wording if you edit it.
- IP102 appears as metrics only, for a 10-class subset; do not describe it as all 102 classes.
  Privacy and compression results are BloodMNIST only.
- Free hosts put apps to sleep when idle, so the first visit may take a minute.
