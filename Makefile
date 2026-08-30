PY ?= python3.12
SEASONS ?= 2022 2023 2024 2025 2026

.PHONY: help install data panel models clean

help:
	@echo "make install   install python dependencies"
	@echo "make data      download nflverse data for seasons: $(SEASONS)"
	@echo "               FORCE=1 make data re-downloads files already on disk"
	@echo "make panel     build data/processed/panel.csv from data/raw/"
	@echo "make models    fit and persist models/{pos}.joblib + MODEL_CARD.md"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.fetch $(SEASONS)

panel:
	$(PY) -m src.features $(SEASONS)

models:
	$(PY) -m src.models

clean:
	rm -rf data/raw/*.csv data/raw/*.csv.gz data/processed/*
