# Action plan

Goal unchanged: compare feature extractors and similarity measures for image retrieval.
Phase 1 fixes what makes current results wrong. Phase 2 modernizes only after Phase 1 results are regenerated.

## Phase 1 — Correctness

### Retrieval and metrics
- [x] Cosine ranking: search returns least similar images (`src/addons/finder.py:116,150`). Use a distance (`pairwise_distances(metric="cosine")`) so every measure sorts ascending.
- [x] Replace `Finder` and its three subclasses with `sklearn.metrics.pairwise_distances(metric=...)`.
- [x] AP/MAP normalized by retrieved hits instead of `min(R, k)` (`src/addons/metrics.py:111-114`). Pass the relevant count `R`.
- [x] Rename reported metrics to MAP@5 / MRR@5; record `k` with evaluation output.
- [x] `zip(..., strict=True)` in aggregate metrics; define empty-input behavior.
- [x] Rewrite `tests/test_metrics.py` for the corrected AP definition, rank beyond `k`, no-hit queries, and ordering for all three measures.

### Extraction
- [x] AKAZE/ORB: replace response-sorted concatenation + zero padding with a bag of visual words (vocabulary built on train split only, Hamming assignment for binary descriptors).
- [x] CNN grayscale input: `decode_jpeg(..., channels=3)`, drop the dead rank check (`src/addons/extraction/compressor.py:51-53`).
- [x] Use `include_top=False, pooling="avg"` for NasNet/EfficientNet; keep VGG `fc2` and document the chosen layer per model.
- [x] Batch CNN inference when building the gallery; `model(x, training=False)` for single images.
- [x] Delete dead code: `SIFTDescriptor`, `Finder.change_database`, `timeit`, `preprocess` in the `Extractor` protocol.

### Evaluation pipeline
- [x] Store `image_id`/path and `split` with every vector; fixed-width `pl.Array(pl.Float32, dim)` column.
- [x] Extract query (test) features once per extractor instead of once per measure; load each model once.
- [x] Time extraction and search separately, with a warmup call; state which statistic (median/mean) is reported.
- [x] Record extraction failures (no keypoints) and gallery size per extractor; pick one policy (common subset or count failures).
- [x] Sort image paths before the split; fix docstring ratio (67/33, not 70/30).

### Results and docs
- [x] Re-run features + evaluation; regenerate notebook and `reports/1.0-evaluation.pdf`.
- [x] Regenerate README results table from the notebook; remove the "cosine performs poorly" claim unless the new results support it.
- [x] Fix dataset counts in the notebook (11385 vs 10833 + 5337).

### Makefile and setup
- [x] `make prepare`: create `.env` only if absent (currently erases Kaggle credentials).
- [x] `make notebook`: `cd notebooks/ &` runs Jupyter from the root; fix and add to `.PHONY`.
- [x] README: add the missing `make build` step; `src/data/` → `src/dataset/`.

## Phase 2 — Modernization

### Python and dependencies
- [ ] Python version: currently 3.12 (`requires-python = ">=3.12"`, `uv venv --python 3.12`). The repo pins `tensorflow>=2.18,<2.19`, citing `tensorflow-metal` 1.2.0 compatibility; PyPI states 1.2.0 targets "2.18+ TF versions", so the pin is a repo choice, not a proven ceiling. Verify which TF / tensorflow-metal / PyTorch releases work on 3.13+, then raise the floor and pin an upper bound that is actually tested.
- [x] Keep one OpenCV package (`opencv-python` and `opencv-contrib-python` conflict). Done in Phase 1: OpenCV 5 moved AKAZE to contrib, so only `opencv-contrib-python>=5.0` remains.
- [ ] Commit `uv.lock` (remove from `.gitignore`); Makefile uses `uv sync` / `uv run`.
- [ ] `tensorflow-metal` in a macOS-only extra; JupyterLab in a `notebook` extra.
- [ ] Replace black + isort + pylint + radon with ruff; delete `pylintrc`.

### Storage
- [ ] Keep Parquet + exact brute-force search as the baseline. Chroma is rejected as main storage (no L1 distance, approximate HNSW search).
- [x] Benchmark exact search on real dimensions for all three measures. Measured in Phase 1 (7,627 gallery images, M1 Pro, amortized per query): cosine and Euclidean under 1 ms, Manhattan 11–18 ms for the networks.
- [ ] Optional: compare exact search against an approximate index (FAISS HNSW or Chroma) with recall@k.

### Extractors
- [ ] Add DINOv2 (self-supervised).
- [ ] Add OpenCLIP or SigLIP2 (image-text pretraining).
- [ ] Add a modest modern CNN (ConvNeXt-Tiny).
- [ ] Decide TensorFlow vs PyTorch (MPS) after reproducing legacy results; measure, don't assume.
