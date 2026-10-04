PY ?= python3.11
VENV ?= .venv
PIP = $(VENV)/bin/python -m pip
PYTHON = $(VENV)/bin/python
CONFIG ?= config.yaml

.PHONY: env fixture test bootstrap t01 t02 t03 t05

env:
	test -d $(VENV) || $(PY) -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e .
	$(PYTHON) -m pip freeze > requirements.txt

bootstrap:
	$(PYTHON) scripts/t00_bootstrap_sample.py --out data/derived/bootstrap --phresh 100 --putra 25

# Build data/fixture/ (20 fixed ids in data/fixture/ids.txt) from any Putra copy.
fixture:
	$(PYTHON) scripts/make_fixture.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

t01:
	$(PYTHON) scripts/t01_env_and_data.py --config $(CONFIG)
t02:
	$(PYTHON) scripts/t02_states.py --config $(CONFIG)
t03:
	$(PYTHON) scripts/t03_jev_smoke.py --config $(CONFIG)
t05:
	$(PYTHON) scripts/t05_determinism.py --config $(CONFIG) --light
