export PYTHONPATH=/Users/schalappe/Documents/Lab/Engineer/Projects/content-based-image-retrieval

VIRTUAL_ENV=venv
PYTHON=${VIRTUAL_ENV}/bin/python
PIP=${VIRTUAL_ENV}/bin/pip

.PHONY: build features predict

venv: requirements.txt
	python3 -m venv venv
	$(PIP) install -r requirements.txt

prepare:
	mkdir -p data/inputs
	mkdir -p data/features
	mkdir -p data/evaluation
	@echo RAW_PATH= >> .envvv
	@echo INPUT_PATH=$(shell pwd)/data/inputs >> .envvv
	@echo FEATURE_PATH=$(shell pwd)/data/features >> .envvv
	@echo EVALUATION_PATH=$(shell pwd)/data/evaluation >> .envvv


build:
	$(PYTHON) src/data/build_dataset.py

features:
	$(PYTHON) src/features/build_features.py

predict:
	$(PYTHON) src/models/make_prediction.py