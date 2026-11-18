# CLAUDE.md: FedSurveil coding rules

- Follow PLAN.md; if anything conflicts, PLAN.md wins. Work one step at a time.
- Python 3.11. Plain PyTorch for the federated simulator; Opacus for DP-SGD; Flower only for the single end-to-end demo run. No other FL framework.
- Use group norm, never batch norm (it breaks under non-IID data and under DP).
- Do not use DermaMNIST (CC-BY-NC). Do not redistribute IP102 or show its images in the public dashboard; metrics only.
- No secret is required. Never commit `.env`; `.env.example` has names with empty values.
- Privacy claims must say which guarantee they give (sample-level vs client-level DP). Secure aggregation is simulated: never call it secure or cryptographic in code, docs or the UI without saying so.
- Training runs in Kaggle/Colab notebooks (or locally) and writes JSONL logs to `runs/`; analysis and the dashboard run from logs. CI installs no torch.
- Fixed seeds. Every run records dataset, algorithm, alpha, number of sites, DP settings, seed, library versions and git commit in `results/run_meta.json` or its run log.
- Format and lint with ruff (line length 100). Write tests in `tests/` for every feature; run `pytest` before each commit.
- Code comments only for non-obvious reasons. Plain-English explanations go in LEARNING.md, not chat.
- Commit with the step name in the message, then push (if a remote exists).
