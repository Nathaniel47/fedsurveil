# LEARNING.md: FedSurveil

Plain-English notes on what was built and why it matters. One section per step.

## Step 2: Repo setup

**What was built.** A working but empty project: a Python package `fedsurveil/` with one folder per part of the plan (data, models, federated learning, privacy, communication, Flower demo, evaluation), a `config.py` holding dataset, seed and privacy choices, dependency files, a Dockerfile, formatting and secret checks, a placeholder Streamlit dashboard, and a GitHub Actions workflow that runs the tests.

**Why the dependencies are split in three.**
- `requirements.txt` is what the dashboard needs (Streamlit, Plotly, numpy, pandas). It has no PyTorch, so the dashboard deploys on free hosting and CI stays fast.
- `requirements-ml.txt` adds PyTorch, Opacus, Flower and MedMNIST. You install it only where you train (Kaggle, Colab or a GPU machine).
- `requirements-dev.txt` adds pytest, ruff and pre-commit.

Training saves logs to files and the dashboard only reads them. That is why a lost notebook session costs nothing and the public app needs no GPU.

**Two choices written into the config and tests.** DermaMNIST is left out because its licence forbids commercial use, and a test fails if someone adds it. `medmnist` is pinned below 3.0 because version 3 changed the available image sizes.

**Secrets.** FedSurveil needs no secret. `.env` is still gitignored, `.env.example` lists names with empty values (a test checks that), and the gitleaks hook blocks commits that look like they contain a key.

**How to check it works.** Create a virtual environment, run `pip install -r requirements-dev.txt`, then `pytest` and `ruff check .`. All should pass.

**Reading.** The pre-commit documentation (pre-commit.com) intro page: hooks in five minutes.
