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

## Step 4: Core feature (non-IID sites and FedAvg)

**What was built.** The simulator for "5-20 sites with realistic class imbalance". `python -m fedsurveil.fl.simulate` takes a dataset, a number of sites and a Dirichlet alpha, and runs either FedAvg across the sites or a central baseline that trains one model on all the pooled images. Each round it logs global accuracy, macro-F1, the accuracy at every site, and the bytes sent, to a JSONL file in `runs/`. The first line of each log records the settings, library versions and git commit, so any result can be traced back.

**How one FedAvg round works.**
1. The server sends the global model to the chosen sites.
2. Each site trains its own copy on its own images for a few epochs. Nothing but the model leaves the site.
3. Each site sends back the *change* it made (its new weights minus the global weights).
4. The server adds the average of the changes to the global model, weighting each site by how many images it has. A site with 4x the data counts 4x.

**Why per-site accuracy matters.** One global accuracy hides unfairness: a model can score 90% overall and fail at the one small clinic whose patients look different. So each site gets its own evaluation set drawn from the global test images, with the same class mix as that site's training data. The log reports the mean, the worst site and the spread.

**Why group norm, not batch norm.** Batch norm computes statistics from each training batch. When sites hold different classes, those statistics differ a lot between sites, and averaging them gives a muddled model. Group norm works on one image at a time, so it does not have this problem (and it is also needed later for differential privacy). A test checks that no batch-norm layer slips in.

**Why the central baseline.** It is the best case: one model that sees everything. The gap between it and FedAvg is the cost of keeping data at the sites. A fair comparison gives both the same number of passes over the data per round.

**Why bytes are counted.** Every round, every chosen site downloads and uploads a full model. The small CNN is about 379 KB, so 10 sites cost about 7.6 MB per round (3.8 MB down, 3.8 MB up). This is measured from the real weights, not estimated, and later steps compare it with compressed updates.

**What was checked.** 19 new tests (35 in total, pytest). The aggregation maths is checked by hand (a size-weighted mean, and one site gives back exactly that site's model). Training is deterministic for a fixed seed, and two runs with the same seed log identical numbers. On a toy dataset the model learns, the byte counts match the number of sites chosen, and the baseline sends no traffic. A one-minute CPU run on 400 real BloodMNIST images confirmed the real data path and log format work. Its accuracy (about 19%) means nothing: it only proves the plumbing.

**Result of the check (BloodMNIST, 3 seeds).** The experiment ran on a free Colab T4 and non-IID data clearly hurts: central training reaches about 94.8%, FedAvg on near-IID sites (alpha 100) 90.3%, alpha 0.5 reaches 87.5% (but its worst site is only 76%), and alpha 0.1 reaches only about 60% with a worst site near 10%. This held in all 3 seeds. Details and caveats (3 seeds, 30 rounds, untuned) are in `results/step4_noniid_check.md`. PathMNIST and IP102 are not run yet.

**How to run and check it.** `pytest` for the tests. For the experiment, open the notebook on Kaggle with the GPU on, run it once with `SMOKE = True`, then set it to `False`. It zips the logs in `runs/`.

**Reading.**
- McMahan et al. 2017, "Communication-Efficient Learning of Deep Networks from Decentralized Data" (arXiv): the FedAvg paper.
- Hsu et al. 2019, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification" (arXiv): why non-IID data hurts and how the Dirichlet split models it.
- Wu and He 2018, "Group Normalization" (arXiv): why it replaces batch norm here.
