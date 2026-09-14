# Self-inoculation toy: code and design documents

Companion material for the post *Self-Inoculation* (asving.com/blog/self-inoculation/).

- `PREREG.md` — preregistered design, predictions and falsifiers, with dated amendments (including the
  adversarial-review round and the one prediction that failed).
- `RESULTS.md` — every number behind the post, per seed, for both parameterizations of the shared circuit.
- `process.py` — the generative process (Mess3 HMMs, non-ergodic case stream, exact Bayesian filters, the
  target swap) · `model.py`, `train.py` — the 4-block GPT and the three training phases ·
  `probe.py`, `restore_calib.py`, `meandiff_test.py` — probes, encoder-image erasure, situation patch,
  weight restoration, calibration, mean-difference test · `analyze.py`, `blog_figs*.py`, `oracle_sim.py`,
  `scan_params.py` — figures and the CPU oracle / parameter scan.
- `reviews/` — the three adversarial reviews of the design and write-up by a different model (Codex), which
  the amendments in PREREG.md and the wording in RESULTS.md respond to.

Environment: Python with numpy, torch, scipy, matplotlib. The toy trains in minutes on one GPU or in an
hour or two on CPU (`TORCH_THREADS=4`).
