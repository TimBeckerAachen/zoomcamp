# docker build -t flower-predict .
# docker run -it -p 9696:9696 --name test_flowers flower-predict
FROM python:3.9-slim-bullseye
ARG COPY_DIR="./"

WORKDIR "./"

COPY ["${COPY_DIR}requirements.txt", "./"]
COPY ["${COPY_DIR}predict.py", "./"]
COPY ["${COPY_DIR}model.tflite", "./"]

RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ -r requirements.txt

EXPOSE 9696

ENTRYPOINT ["gunicorn", "predict:app"]
