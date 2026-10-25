# telltail: prototype sandbox for physics-based anomaly metrics (LANL auth logs).
# Run `make help` to list targets.

PY := .venv/bin/python
PIP := .venv/bin/pip
PYTHON_MIN := 3.11  # floor; developed/tested on 3.14 (see .python-version)
DATA := data/lanl/derived/compromised_users.csv  # labeled auth subset scored by every method

.PHONY: help check-python setup test data baselines eval sweep plots clean clean-derived

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

check-python:  ## fail unless python3 is >= 3.11
	@python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' || \
		{ echo "Python $(PYTHON_MIN)+ required, found $$(python3 -V 2>&1)"; exit 1; }

setup: check-python .venv  ## create venv and install pinned deps

.venv: requirements.txt
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch .venv

test: setup  ## run unit tests
	$(PIP) install -r requirements-dev.txt
	$(PY) -m pytest tests/ -q

data: setup  ## build labeled auth subsets under data/lanl/derived/
	$(PY) src/make_subsets.py

baselines: data  ## score every method, write outputs/*_alerts.csv
	$(PY) src/baseline_novelty.py $(DATA)
	$(PY) src/baseline_iforest.py $(DATA)
	$(PY) src/surprisal.py $(DATA)

eval: baselines  ## recall@budget table per method
	$(PY) src/evaluate.py outputs/novelty_alerts.csv --events $(DATA)
	$(PY) src/evaluate.py outputs/iforest_alerts.csv --events $(DATA)
	$(PY) src/evaluate.py outputs/surprisal_alerts.csv --events $(DATA)

sweep: setup  ## every method x standardization row of the README table
	@test -f $(DATA) || { echo "$(DATA) missing, run: make data"; exit 1; }
	@mkdir -p outputs/sweep
	@bash -o pipefail -c '$(PY) src/sweep_observables.py --input $(DATA) \
		2>&1 | tee outputs/sweep/sweep.out'

plots: eval  ## recall-vs-budget figure -> outputs/recall_vs_budget.png
	$(PY) src/plots.py

clean-derived:  ## remove generated subsets, keep raw data
	rm -rf data/lanl/derived

clean: clean-derived  ## also remove venv and python caches
	rm -rf .venv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
