export PYTHONPATH=/Users/schalappe/Documents/Lab/Engineer/Projects/content-based-image-retrieval

VIRTUAL_ENV=venv
PYTHON=${VIRTUAL_ENV}/bin/python
PIP=${VIRTUAL_ENV}/bin/pip

.PHONY: build

venv: requirements.txt
	python3 -m venv venv
	$(PIP) install -r requirements.txt

build:
	$(PYTHON) src/data/build_dataset.py

features:
	$(PYTHON) src/features/build_features.py
