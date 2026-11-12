PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install lint format test parity data artifacts predict wheel clean

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e '.[dev,seq,package,viz]'
	$(BIN)/pre-commit install

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:
	$(BIN)/pytest

parity:
	$(BIN)/pytest -m parity

data:
	bash scripts/fetch_data.sh

artifacts:
	bash scripts/fetch_artifacts.sh

predict:
	$(BIN)/rogii predict --profile vp_balanced_modelpkg_005 --structural v5

wheel:
	bash scripts/build_kaggle_dataset.sh

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
