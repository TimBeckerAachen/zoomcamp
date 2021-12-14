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
  - 764 images of daisies
  - 784 images of roses
  - 984 images of tulips
  - 1052 images of dandelions
  - 733 images of sunflowers
- pictures are divided into five classes: chamomile, tulip, rose, sunflower, dandelion
- photos have different sizes

<p float="left">
    <figure>
        <img src="example_images/rose.jpg" alt="rose" style="width:20%">
        <figcaption>Rose</figcaption>
    </figure>
    <figure>
        <img src="example_images/daisy.jpg" alt="daisy" style="width:20%">
        <figcaption>Daisy</figcaption>
    </figure>
    <figure>
        <img src="example_images/sunflower.jpg" alt="sunflower" style="width:20%">
        <figcaption>Sunflower</figcaption>
    </figure>
    <figure>
        <img src="example_images/tulip.jpg" alt="tulip" style="width:20%">
        <figcaption>Tulip</figcaption>
    </figure>
    <figure>
        <img src="example_images/dandelion.jpg" alt="dandelion" style="width:20%">
        <figcaption>Dandelion</figcaption>
    </figure>
<p>

# notebook
The EDA and model selection can be found in the notebook. The final cells of the notebook contain code to test the local flask app and the public API.

# virtual environ
To create a virtual environment (e.g. with virtualenv) and install the required packages, follow these steps:
1. pip install virtualenv
2. virtualenv <virtual_env_name>
3. source <virtual_env_name>/bin/activate
4. pip install install --extra-index-url https://google-coral.github.io/py-repo/ -r requirements.txt
Leave the environment via:
5. deactivate

# Docker
To run the docker container locally, navigate to the capstone_project folder and execute the commands:
1. docker build -t <image_name> . 
2. docker run --name <container_name> -p 9696:9696 <image_name>

# public APP deployment
The APP is deployed on Heroku. You can send a POST request via `https://flower-types.herokuapp.com/predict` with a json 
body with an url pointing to an image of a flower. The last cell of the notebook provides an example on how to use the APP.

In order to deploy the APP, you have to create an account at `https://www.heroku.com`. After creating an 
account, you can create a new APP by choosing a name and a region. Afterwards, you can deploy your APP by pushing 
docker images to Heroku. Just follow these steps:
1. brew tap heroku/brew && brew install heroku (install heroku cli on MAC `https://devcenter.heroku.com/articles/heroku-cli`)
2. heroku login -i
3. heroku container:login
4. docker tag <image_name> registry.heroku.com/<app-name>/web
5. docker push registry.heroku.com/<app-name>/web
6. heroku container:release web -a <app-name>

# To-do
- different image sizes?
- data augmentation
- error -> not enough images (validation)
- check images that were classified wrongly