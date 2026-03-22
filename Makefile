.PHONY: install download-data eda baseline train evaluate export benchmark report clean all

install:
	pip install -r requirements.txt
	pip install -e .

download-data:
	bash scripts/download_data.sh

eda:
	python scripts/run_eda.py

baseline:
	python scripts/run_baseline.py

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

export:
	python scripts/export.py --config configs/default.yaml

benchmark:
	python src/deployment/benchmark.py

report:
	cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/

all: install download-data eda baseline train evaluate export report
