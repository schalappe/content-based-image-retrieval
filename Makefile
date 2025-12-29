export PYTHONPATH=$(shell pwd)

VIRTUAL_ENV=.venv
PYTHON=${VIRTUAL_ENV}/bin/python
JUPYTER=${VIRTUAL_ENV}/bin/jupyter-lab

.PHONY: prepare download build features predict venv venv-dev

venv:
	uv venv $(VIRTUAL_ENV) --python 3.12
	uv pip install -e .

venv-dev: venv
	uv pip install -e ".[dev]"

prepare:
	@mkdir -p data/raw data/inputs data/features data/evaluation
	@printf '%s\n' \
		"RAW_PATH=$(shell pwd)/data/raw" \
		"INPUT_PATH=$(shell pwd)/data/inputs" \
		"FEATURE_PATH=$(shell pwd)/data/features" \
		"EVALUATION_PATH=$(shell pwd)/data/evaluation" \
		"KAGGLE_USERNAME=" \
		"KAGGLE_KEY=" > .env
	@echo "Created .env file. Add your Kaggle credentials (https://www.kaggle.com/settings > API)"

download:
	$(PYTHON) src/dataset/download_dataset.py

build:
	$(PYTHON) src/dataset/build_dataset.py

features:
	$(PYTHON) src/features/build_features.py

predict:
	$(PYTHON) src/models/make_prediction.py

notebook:
	cd notebooks/ & $(JUPYTER) --port=8080
