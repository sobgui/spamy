venv:
	python -m venv venv

install_requirements:
	.\venv\Scripts\activate
	pip install -r requirements.txt
	deactivate

train:
	.\venv\Scripts\activate
	python src/train.py
	deactivate

predict:
	.\venv\Scripts\activate
	python src/predict.py
	deactivate

test_units:
	.\venv\Scripts\activate
	pytest test/units/ -v
	deactivate

test_performs:
	.\venv\Scripts\activate
	pytest test/performs/ -v
	deactivate

jupyter:
	.\venv\Scripts\activate
	jupyter notebook

streamlit:
	.\venv\Scripts\activate
	streamlit run src/app.py