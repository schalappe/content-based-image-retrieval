export PYTHONPATH=$(shell pwd)

RUN=uv run --frozen

.PHONY: prepare download build features predict notebook venv lint test

venv:
	uv sync --frozen --all-groups

# Never overwrite an existing .env: it holds the Kaggle credentials.
prepare:
	@mkdir -p data/raw data/inputs data/features data/evaluation
	@if [ -f .env ]; then \
		echo ".env already exists, left unchanged."; \
	else \
		printf '%s\n' \
			"RAW_PATH=$(shell pwd)/data/raw" \
			"INPUT_PATH=$(shell pwd)/data/inputs" \
			"FEATURE_PATH=$(shell pwd)/data/features" \
			"EVALUATION_PATH=$(shell pwd)/data/evaluation" \
			"KAGGLE_USERNAME=" \
			"KAGGLE_KEY=" > .env; \
		echo "Created .env file. Add your Kaggle credentials (https://www.kaggle.com/settings > API)"; \
	fi

download:
	$(RUN) src/dataset/download_dataset.py

build:
	$(RUN) src/dataset/build_dataset.py

features:
	$(RUN) src/features/build_features.py

predict:
	$(RUN) src/models/make_prediction.py

notebook:
	cd notebooks && $(RUN) jupyter-lab --port=8080

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

test:
	$(RUN) python -m unittest discover tests
