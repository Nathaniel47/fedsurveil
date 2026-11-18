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

## Step 3: Data pipeline

**What was built.** Four small scripts in `fedsurveil/data/`. `fetch` downloads PathMNIST, BloodMNIST and OrganAMNIST (about 280 MB) and checks each file against the checksum Zenodo publishes. `ip102` turns the manual IP102 download into a small 10-class, 64 px file. `partition` splits a training set across simulated sites. `prepare` saves those splits for every dataset, three alpha values and three seeds (27 splits for the three MedMNIST sets). `report` writes `results/data_report.md`.

**What "non-IID" means here.** In real federated learning each site sees a different slice of the world: one clinic sees mostly one disease. We imitate that with a Dirichlet distribution. A small alpha (0.1) gives each site a few dominant classes (on PathMNIST the biggest class is on average 66% of a site's images). A large alpha (100) makes every site look like the whole dataset (15%). Site sizes also differ (smallest site 4,356 images, largest 15,464 at alpha 100 on PathMNIST), because real sites are never the same size.

**Why only the training set is split.** Validation and test sets stay whole and global. Every method (FedAvg, FedProx, SCAFFOLD and the central baseline) is scored on the same test images, so the comparison is fair.

**Why the repo holds hashes, not indices.** A split is rebuilt from its seed in seconds. `data/split_digests.json` stores a short hash per split so you can check you rebuilt the same one, without committing hundreds of thousands of numbers.

**Problems the report found.** BloodMNIST has 8 duplicate training images and 5 training images that also appear in the test set. That is a small leak (5 of 3,421 test images, 0.15%), but it is real, so it is recorded rather than hidden, and results on BloodMNIST should be read with it in mind. PathMNIST and OrganAMNIST showed no overlap. OrganAMNIST has a 4.5x gap between its largest and smallest class, so macro-F1 is reported next to accuracy.

**IP102 added.** The official download (75,222 images, 102 classes) was extracted, and the builder kept the 10 classes with the most training images and resized them to 64 px: 17,926 train, 2,984 val and 8,972 test images (a 300 MB file, gitignored). The class mix is uneven (4.1x between the largest and smallest class) and there is 1 duplicate training image, so macro-F1 is reported too. With the 3 MedMNIST sets that makes 36 saved splits. At alpha 0.1 the smallest of the 10 sites holds only 40 images, which is a deliberately extreme case.

**How to check it.** `pytest` (16 tests: splits are disjoint, cover every image, repeat exactly for the same seed, get more skewed as alpha falls, and detect a corrupted file), then the three commands in the README.

**Reading.** Hsu et al. 2019, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification" (arXiv): the paper that introduced this Dirichlet split.
