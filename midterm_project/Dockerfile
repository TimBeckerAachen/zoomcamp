FROM python:3.9-slim-bullseye

RUN pip install pipenv

WORKDIR "./"

COPY ["Pipfile", "Pipfile.lock", "./"]
COPY ["best_model.bin", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]
COPY ["train.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
