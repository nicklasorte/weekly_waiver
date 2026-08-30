PY ?= python3.12
SEASONS ?= 2022 2023 2024 2025 2026

.PHONY: help install data panel models weekly test clean

help:
	@echo "make install   install python dependencies"
	@echo "make data      download nflverse data for seasons: $(SEASONS)"
	@echo "               FORCE=1 make data re-downloads files already on disk"
	@echo "make panel     build data/processed/panel.csv from data/raw/"
	@echo "make models    fit and persist models/{pos}.joblib + MODEL_CARD.md"
	@echo "make weekly    score the wire pool: make weekly SEASON=2025 WEEK=8"
	@echo "               omit SEASON/WEEK to resolve them from the schedule"
	@echo "make test      run unit tests"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.fetch $(SEASONS)

panel:
	$(PY) -m src.features $(SEASONS)

models:
	$(PY) -m src.models

weekly:
	$(PY) -m src.weekly $(if $(SEASON),--season $(SEASON)) $(if $(WEEK),--week $(WEEK))

test:
	$(PY) -m unittest discover -s tests

clean:
	rm -rf data/raw/*.csv data/raw/*.csv.gz data/processed/*
