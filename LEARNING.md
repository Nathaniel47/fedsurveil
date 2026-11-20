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

## Step 5.1: FedProx and SCAFFOLD

**What was built.** Two more ways to fight client drift, selectable with `--algorithm fedprox --prox-mu 0.1` or `--algorithm scaffold`, next to FedAvg and the central baseline.

**The problem they address.** When a site holds mostly one class, its local training pulls the model toward "always predict that class". Averaging such models gives a worse model than any sensible compromise. Step 4's check showed how bad this gets (alpha 0.1: about 60% vs 95% central).

**FedProx.** Adds a spring: during local training, the loss also penalises moving away from the global model, with strength `mu`. The bigger `mu`, the less each site can wander. Too big and the sites barely learn anything new, so `mu` has to be tuned (the notebook tries 0.01, 0.1 and 1.0). Cost: no extra communication.

**SCAFFOLD.** Each site keeps a "control variate", an estimate of the direction its own data pushes the model. The server keeps the average of these. During local training each gradient is corrected by (server average - this site's estimate), which cancels the part of the push that comes from the site's skewed data. After training, the site updates its estimate from how far its model actually moved. Cost: every round sends the control variates too, so SCAFFOLD **uses twice the bandwidth** of FedAvg (the byte counter reflects this).

**Two honest simplifications.**
- SCAFFOLD's formula is derived for plain SGD. With momentum 0.9 I use the long-run step size `lr / (1 - momentum)` in the formula, which is an approximation. If SCAFFOLD misbehaves in the GPU runs, run it with momentum 0 to check.
- FedProx and SCAFFOLD are compared at the settings tuned for FedAvg, not tuned separately.

**What was checked (15 new tests).** FedProx with a large `mu` ends closer to the starting weights than without it, and `mu = 0` changes nothing. SCAFFOLD with all-zero control variates equals plain training, the control-variate update matches the paper's formula on a hand-worked number, and **a single-site SCAFFOLD run gives exactly the same accuracies as FedAvg**, which is what the maths says must happen. SCAFFOLD sends exactly twice the bytes. Both new methods learn a toy task.

**Not done yet.** The real comparison (FedAvg vs FedProx vs SCAFFOLD vs central on BloodMNIST) needs a GPU: `notebooks/02_fedprox_scaffold.ipynb` runs it. There are no accuracy results for FedProx or SCAFFOLD yet.

**How to run it.** `pytest tests/test_algorithms.py`, then open the notebook on Colab with a T4.

**Reading.**
- Li et al. 2020, "Federated Optimization in Heterogeneous Networks" (arXiv): FedProx.
- Karimireddy et al. 2020, "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (arXiv).

## Step 5.2: Differential privacy and secure aggregation

**What was built.** Three optional privacy layers for the federated runs, plus a way to measure leakage. All are switches on the same command: `--dp-clip 1.0 --dp-noise 1.0` (client-level DP), `--site-dp-noise 1.0` (sample-level DP-SGD), `--secagg` (simulated secure aggregation) and `--mia` (membership-inference check).

**Two different promises, easy to mix up.**
- **Client-level DP** (`--dp-clip`, `--dp-noise`) protects a whole *site*. Someone looking at the final model cannot tell whether clinic X took part in training at all. Each site's update is clipped to a maximum length C, the server averages the clipped updates with equal weight, and adds random noise sized to hide any one site's contribution.
- **Sample-level DP-SGD** (`--site-dp-noise`, uses Opacus) protects a single *image* inside a site: each image's gradient is clipped and noise is added at every training step. It says nothing about whether the site took part.
Reports and charts always say which one a number is about.

**What epsilon means.** Epsilon is the privacy "budget" spent: smaller is more private, and every round spends more of it. The code logs epsilon each round (with delta fixed at 1e-5). Roughly, an epsilon around 1 is strong protection and values in the tens are weak. More noise gives a smaller epsilon but a less accurate model, which is the privacy-accuracy trade-off the GPU notebook measures. Client-level epsilon uses an exact formula when every site takes part each round and Opacus' accountant when only some do. In the partial case it is an approximation, because the accountant assumes each site is picked independently, while we pick a fixed number.

**Secure aggregation, simulated.** Each pair of sites shares a random mask. One site adds it and the other subtracts it, so each site's message on its own looks like random numbers, but the masks cancel when the server adds everything up, and the server learns only the sum. This is a **simulation on one machine, not real cryptography**: no key exchange, no authentication, and no handling of a site dropping out mid-round. What it does show truthfully is the trick and its cost (each site sends a secret to every other site, so the extra traffic grows with the square of the number of sites). The arithmetic is on whole numbers so the masks cancel exactly.

**How the pieces fit.** The order each round is: clip, equal-weight average (done through the masked sum if `--secagg` is on), then noise. Adding the noise after the secure sum is the natural arrangement: the server only ever sees the total, and the noise protects the total.

**Membership-inference check.** A simple attack guesses that an image was in the training set when the model's loss on it is low. Its AUROC is 0.5 if the attack is no better than chance. It is a measurement of our own model on public benchmark data. A low value does not prove privacy, only that this simple attack found nothing, and DP is what gives the actual guarantee.

**Choices and limits to remember.**
- SCAFFOLD is not combined with DP: its control variates are extra data that would need their own privacy analysis.
- Secure aggregation and update compression are not combined: masked numbers are dense random values and cannot be compressed or sparsified.
- Opacus' secure random numbers are turned off for speed, as it warns. Fine for experiments, not for production.
- The noise and clip settings have not been tuned yet. The clip value C is chosen by the notebook from the typical update size.

**What was checked (24 new tests).** Clipping shortens long updates and leaves short ones alone. The noise has exactly the intended spread (measured on a large vector). The closed-form epsilon agrees with Opacus' accountant to within 5%, more noise gives a smaller epsilon and more rounds a bigger one, and partial participation costs less than full. Masks cancel *exactly* in whole-number arithmetic, a single masked update has no measurable correlation with the real one, and the secure sum equals the plain sum. Without any privacy options the pipeline equals plain FedAvg. DP-SGD really changes the model weights and logs its accountant history. Invalid combinations are rejected.

**Not done yet.** No accuracy-versus-epsilon results exist: `notebooks/03_privacy.ipynb` runs the grid on a GPU (three noise levels, DP-SGD, secure aggregation, membership check on every run).

**How to run it.** `pytest tests/test_privacy.py`, then the notebook on a Colab T4.

**Reading.**
- Abadi et al. 2016, "Deep Learning with Differential Privacy" (arXiv): DP-SGD.
- McMahan et al. 2018, "Learning Differentially Private Recurrent Language Models" (arXiv): client-level DP for federated averaging.
- Bonawitz et al. 2017, "Practical Secure Aggregation for Privacy-Preserving Machine Learning".
- Yeom et al. 2018, "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting" (arXiv): the loss-threshold attack.

## Step 5.3: Training dashboard with per-site accuracy

**What was built.** A Streamlit dashboard (`streamlit run app/dashboard.py`) that replays logged runs. It trains nothing, so it runs on any laptop or free host with no GPU. It reads two kinds of files: full logs in `runs/` (written by the simulator) and compact result files in `results/` (the numbers copied from the Colab runs). The loading code lives in `fedsurveil/eval/runs.py` and uses no torch, so it is tested in CI.

**What you can see.**
- **Training:** global accuracy and macro-F1 for any set of runs, a slider to replay up to a given round, a "Play replay" button that animates the accuracy curves as if training were live, and "rounds to reach X% accuracy".
- **Per-site accuracy:** worst-site and mean-site curves, a site-by-round heatmap (full logs only), and final accuracy per site next to how many images each site holds. This is the view that shows a model failing one clinic while the global number looks fine.
- **Data split:** a site-by-class heatmap of the Dirichlet split, so you can see what "non-IID" looks like for the run.
- **Privacy and cost:** accuracy plotted against total megabytes sent, and epsilon over rounds, labelled client-level or sample-level.
- **Summary table:** one row per run, with accuracy, macro-F1, worst site, megabytes, epsilon and the membership-inference score where they exist.

**Why "live" means replay.** A free host has no GPU and cannot train a model while you watch. Replaying a saved log round by round looks the same to a viewer and costs nothing to host, which is the plan from PLAN.md.

**What the dashboard shows right now.** Only the 4 BloodMNIST runs from seed 0 (central and FedAvg at 3 alphas), because those are the only results saved so far. Per-round per-site accuracy was not saved for them (only the worst, mean and final per-site numbers), so the site-by-round heatmap stays empty until runs are re-saved as full logs. Nothing is invented to fill gaps. Raw IP102 images are never shown.

**What was checked (12 new tests).** The loader reads the log format and the compact format, labels runs with their privacy options, prefers a full log over a compact copy of the same run, skips broken or empty files, and counts megabytes and rounds-to-target correctly. A headless run of the dashboard on the committed results finishes without errors. **I have not looked at the page in a browser**, so the layout and the replay animation are untested by eye.

**How to run it.** `streamlit run app/dashboard.py`, or `docker compose up`.

**Reading.** The Streamlit documentation page on caching (`st.cache_data`): why the files are loaded once and not on every click.

## Step 5.4: Communication cost for low-bandwidth sites

**Why it matters.** A clinic or field site in a low-bandwidth area may have a link of 0.1 to 1 Mbit/s. The small CNN here is 379 KB, so one dense update takes about 30 seconds to upload on a 0.1 Mbit/s link, and every site has to do that every round. The number of rounds, and the size of each upload, decide whether federated learning is practical there at all.

**What was built.**
- **Compression** (`--compress int8` or `--compress topk --topk-fraction 0.1`): `int8` stores each weight in one byte plus one shared scale, about 4x smaller. `topk` sends only the largest 10% (or 1%) of the weights, each with its position, so 10% costs about 20% of the dense size (5x smaller) and 1% about 50x smaller. The simulator logs the compressed bytes each round, so the cost is measured, not guessed.
- **Error feedback** for top-k: what is dropped this round is remembered at the site and added to the next round's update, so nothing is lost for good, only delayed. Without it top-k usually hurts accuracy badly. A test checks that sent plus remembered always equals everything put in.
- **A report** (`python -m fedsurveil.eval.comms_report`, writes `results/comms_report.md`) with two parts: the exact bytes one site uploads under each scheme, and for every logged run the rounds, megabytes and hours to reach 80% and 85% accuracy on 2G, 3G and 4G-speed links.

**What the numbers say so far (BloodMNIST, seed 0, no compression).** Reaching 80% accuracy took 11 rounds (83 MB in total) with near-IID sites and 13 rounds (99 MB) at alpha 0.5. On a 0.1 Mbit/s link that is roughly 0.2 hours of transfer per site, and at alpha 0.1 the run never reached 80%. Non-IID data costs bandwidth as well as accuracy. SCAFFOLD sends twice as much per round as FedAvg.

**Honest limits.**
- The link speeds are illustrative assumptions (sites work in parallel, a fixed latency per transfer, no protocol overhead), not measurements of any real network.
- **No compressed run has been trained yet.** The sizes in the report are exact, but whether int8 or top-k costs accuracy or extra rounds is unmeasured until `notebooks/04_compression.ipynb` runs on a GPU.
- Compression is applied to the model update only (not to the download of the global model), and it is not combined with secure aggregation (masked numbers cannot be sparsified) or SCAFFOLD. With client-level DP the update is compressed first and clipped afterwards, which keeps the privacy bound valid.

**What was checked (18 new tests).** int8 error is at most half a quantization step, the byte counts match the formulas exactly, top-k keeps exactly the largest entries (and at least one), error feedback conserves the total update, and an int8 run in the simulator uploads about a quarter of the dense bytes and still learns a toy task. The report's time arithmetic is checked on a hand-worked example.

**How to run it.** `pytest tests/test_compress.py tests/test_comms.py`, `python -m fedsurveil.eval.comms_report`, then the notebook on a Colab T4.

**Reading.**
- Konecny et al. 2016, "Federated Learning: Strategies for Improving Communication Efficiency" (arXiv).
- Stich et al. 2018, "Sparsified SGD with Memory" (arXiv): why error feedback works.

## Step 6: The Flower end-to-end demo

**Why it matters.** Every experiment so far ran in one Python process: a loop that pretends to be many sites. That is fast and easy to debug, but it never sends a model over a connection. The demo runs the same idea the way a real deployment would, with separate programs that talk over the network, so you can show the real protocol and not only a simulation.

**What was built.** Three small pieces in `fedsurveil/flower_demo/` and one launcher.
- A **SuperLink** is the hub. A **SuperNode** is one site's agent. Each SuperNode runs a **ClientApp** (`client.py`): it receives the global model, trains on its own share of the data and sends the new weights back. The **ServerApp** (`server.py`) averages the weights with FedAvg and logs the global test accuracy after every round.
- `common.py` is shared. A site works out its own images from its site number, the number of sites, alpha and the seed, so no images ever travel, only model weights.
- `scripts/run_flower_demo.py` starts the SuperLink and three SuperNodes, submits the app with `flwr run`, then stops everything. The settings are in `flower_app/pyproject.toml`.

**What the run shows.** Five rounds on 3,000 BloodMNIST images split over 3 sites: global accuracy goes 17%, 20%, 19%, 16%, 41%, 58% (round 0 is the untrained model). It is a protocol demo on a small subset, not a result: it carries no privacy, no comparison and no seeds.

**Honest limits.**
- No TLS and no SuperNode authentication, all on localhost: a demo, not a secure deployment.
- It runs plain FedAvg only; FedProx, SCAFFOLD, DP and compression exist only in the simulator.
- Flower rewrote parts of its own setup on first run: it created `~/.flwr/config.toml` and installed a copy of the app under `~/.flwr/apps/`. Both are outside the repo.
- The result goes to `results/flower_demo.json`, not `runs/`, because it lacks the per-site accuracies the dashboard expects.

**What was checked (4 new tests).** Sites hold disjoint images and get the same ones on every run, the subset is seeded, a model survives the trip through Flower's array format unchanged, and a site's data loads from the real dataset. The full run was done by hand, since it needs several processes.

**How to run it.** `pytest tests/test_flower_demo.py`, then `python scripts/run_flower_demo.py`.

**Reading.** The Flower documentation page "Run Flower with deployment engine" and the first page of McMahan et al. 2017, "Communication-Efficient Learning of Deep Networks from Decentralized Data" (arXiv), which defines FedAvg.

## Step 7: Evaluation and what the results mean

**Why it matters.** A pile of run logs is not a result. This step turns them into one table with spreads across seeds, three charts, and a write-up that says what was not shown.

**What was built.** `python -m fedsurveil.eval.report` reads the saved Step 4 and Step 5 result files and writes `results/results_table.md` and three charts. `docs/REPORT.md` is the write-up. The dashboard has a new tab, "Final results", with the same table and charts. `docs/DEPLOY.md` lists the free-hosting steps and `docs/portfolio_card.md` is the card text.

**What the results say (BloodMNIST only).** Skewed sites hurt the worst-off site most. SCAFFOLD helps under strong skew, FedProx barely helps. With only 10 sites, client-level DP with a meaningful epsilon destroys accuracy. Compression to 10x is cheap, 50x costs about 5 points. The simple membership-inference attack found nothing even without DP, which says the attack is weak, not that the model is safe.

**Honest limits.** One dataset and three seeds. The same seed gave results up to 0.7 points apart on two Colab sessions, so differences near one point are noise. PathMNIST, OrganAMNIST and IP102 were never trained, although the data pipeline handles them.

**What was checked (5 new tests).** Every setting has 3 seeds except the single DP-SGD run, baselines match the Step 4 note, every privacy row names its guarantee and the secure-aggregation row says it is a simulation, and the table and charts are written.

**How to run it.** `pytest tests/test_report.py`, then `python -m fedsurveil.eval.report`.

**Reading.** Hoefler and Belli 2015, "Scientific Benchmarking of Parallel Computing Systems" (the rules for reporting variance honestly apply to ML experiments too).
