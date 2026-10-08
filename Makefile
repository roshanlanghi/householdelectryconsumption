.PHONY: install download dashboard test clean

install:
	pip install -r requirements.txt

download:
	python download_data.py

dashboard:
	streamlit run dashboard/app.py

test:
	python -m unittest discover tests/

clean:
	rm -rf __pycache__ .pytest_cache
