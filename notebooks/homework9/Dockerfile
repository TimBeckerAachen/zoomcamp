# docker pull agrigorev/zoomcamp-cats-dogs-lambda:v2
# docker build -t homework9 .
# docker image ls -a
# 322fc756f258
# docker run -it -p 8080:8080 --name test_9 homework9
FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

COPY ["requirements.txt", "./"]
COPY ["lambda_function.py", "./"]

RUN pip install -r requirements.txt
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

CMD [ "lambda_function.lambda_handler" ]
