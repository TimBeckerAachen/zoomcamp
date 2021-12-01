#!/usr/bin/env python
# coding: utf-8

import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import tflite_runtime.interpreter as tflite


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 255.0
    return x


interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = {0: 'cat', 1: 'dog'}
target_size = (150, 150)


def predict(url):
    raw_img = download_image(url)
    img = prepare_image(raw_img, target_size)

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0][0]
    animal = classes[float_predictions > 0.5]

    print(f'It is a {animal} with {float_predictions}')
    return animal


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

