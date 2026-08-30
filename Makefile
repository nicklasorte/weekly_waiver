# Interpreter used to bootstrap the project venv. Must already exist on PATH.
BOOTSTRAP_PY ?= python3.12
VENV ?= .venv

# Interpreter that runs the pipeline. Defaults to the venv `make install`
# builds, so no `activate` step is needed. Override with PY=<python> to use an
# environment you manage yourself, in which case `make install` installs into
# that environment instead of creating a venv.
PY ?= $(VENV)/bin/python

SEASONS ?= 2022 2023 2024 2025 2026

.PHONY: help install data panel models weekly report ledger test clean check-py

help:
	@echo "make install   create $(VENV) and install python dependencies into it"
	@echo "               BOOTSTRAP_PY=python3.x chooses the interpreter it is built from"
	@echo "               PY=<python> installs into that interpreter instead"
	@echo "make data      download nflverse data for seasons: $(SEASONS)"
	@echo "               FORCE=1 make data re-downloads files already on disk"
	@echo "make panel     build data/processed/panel.csv from data/raw/"
	@echo "make models    fit and persist models/{pos}.joblib + MODEL_CARD.md"
	@echo "make weekly    score the wire pool: make weekly SEASON=2025 WEEK=8"
	@echo "               omit SEASON/WEEK to resolve them from the schedule"
	@echo "make report    write the weekly markdown report: make report SEASON=2025 WEEK=8"
	@echo "               add ROSTER=path to resolve drops and the roster check"
	@echo "make ledger    grade logged claims against the naive benchmarks"
	@echo "make test      run unit tests"

# Creating the venv is skipped when PY points somewhere else, so an externally
# managed environment is installed into directly rather than shadowed by .venv.
install:
ifeq ($(PY),$(VENV)/bin/python)
	$(BOOTSTRAP_PY) -m venv $(VENV)
endif
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# Fail with a pointer to `make install` rather than a bare "command not found".
check-py:
	@command -v $(PY) >/dev/null 2>&1 || { \
	  echo "error: no Python interpreter at '$(PY)'"; \
	  echo "  run 'make install' to create it, or pass PY=<python> to use your own."; \
	  exit 1; }

data: check-py
	$(PY) -m src.fetch $(SEASONS)

panel: check-py
	$(PY) -m src.features $(SEASONS)

models: check-py
	$(PY) -m src.models

weekly: check-py
	$(PY) -m src.weekly $(if $(SEASON),--season $(SEASON)) $(if $(WEEK),--week $(WEEK))

report: check-py
	$(PY) -m src.report --season $(SEASON) --week $(WEEK) $(if $(ROSTER),--roster $(ROSTER))

ledger: check-py
	$(PY) -m src.ledger

test: check-py
	$(PY) -m unittest discover -s tests

clean:
	rm -rf data/raw/*.csv data/raw/*.csv.gz data/processed/*
