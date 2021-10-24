FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
COPY ["model1.bin", "dv.bin", "./"]

RUN pipenv install --system --deploy

COPY ["api.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "api:app"]
