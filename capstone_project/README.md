# zoomcamp
Capstone project for ML zoomcamp course: https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp

# link to dataset
Flowers recognition dataset from kaggle:
https://www.kaggle.com/alxmamaev/flowers-recognition

# goal of the capstone project
The aim of the project is to create a model that can classify images of chamomile, tulip, rose, sunflower and dandelion 
by providing urls to relevant pictures to an endpoint. The model should learn to recognize the varying characteristics
from images and to predict the correct flower type.

# Description of the dataset
- dataset contains 4242 images of flowers
- pictures are divided into five classes: chamomile, tulip, rose, sunflower, dandelion
- photos have different sizes

<div class="row">
  <div class="rose">
    <img src="example_images/rose.jpg" alt="rose" style="width:20%">
  </div>
  <div class="daisy">
    <img src="example_images/daisy.jpg" alt="daisy" style="width:20%">
  </div>
  <div class="sunflower">
    <img src="example_images/sunflower.jpg" alt="sunflower" style="width:20%">
  </div>
  <div class="tulip">
    <img src="example_images/tulip.jpg" alt="tulip" style="width:20%">
  </div>
  <div class="dandelion">
    <img src="example_images/dandelion.jpg" alt="dandelion" style="width:20%">
  </div>
</div>

# notebook
The EDA and model selection can be found in the notebook. The final cells of the notebook contain code to test the local flask app and the public API.

# virtual environ
To install the required packages and enter the virtual environment, follow these steps:
1. pipenv install
2. pipenv shell

# Docker
To run the docker container locally, navigate to the midterm_project folder and execute the commands:
1. docker build -t <image_name> . 
2. docker run --name <container_name> -p 9696:9696 <image_name>

# public APP deployment
The APP is deployed on PythonAnywhere. You can send a POST request via `http://Karmufel.pythonanywhere.com/predict`.
The last cell of the notebook provides an example on how to use the APP.

In order to deploy the APP, you have to create an account at `https://www.pythonanywhere.com`. After creating an 
account, you need to upload the model `best_model.bin` and `flask_app.py` the script containing the flask APP to the 
`mysite` folder. Subsequently, you need to open the console and install the required libraries. Finally, you can 
navigate to the `Web` section and click on `Run until 3 month from today`. At the top of the page you can find the url.