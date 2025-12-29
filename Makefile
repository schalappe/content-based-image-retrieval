export PYTHONPATH=$(shell pwd)

VIRTUAL_ENV=.venv
PYTHON=${VIRTUAL_ENV}/bin/python
JUPYTER=${VIRTUAL_ENV}/bin/jupyter-lab

.PHONY: prepare build features predict venv venv-dev

venv:
	uv venv $(VIRTUAL_ENV) --python 3.12
	uv pip install -e .

venv-dev:
	uv venv $(VIRTUAL_ENV) --python 3.12
	uv pip install -e ".[dev]"

prepare:
	mkdir -p data/inputs
	mkdir -p data/features
	mkdir -p data/evaluation
	@echo RAW_PATH= >> .env
	@echo INPUT_PATH=$(shell pwd)/data/inputs >> .env
	@echo FEATURE_PATH=$(shell pwd)/data/features >> .env
	@echo EVALUATION_PATH=$(shell pwd)/data/evaluation >> .env


build:
	$(PYTHON) src/data/build_dataset.py

features:
	$(PYTHON) src/features/build_features.py

predict:
	$(PYTHON) src/models/make_prediction.py

notebook:
	cd notebooks/ & $(JUPYTER) --port=8080