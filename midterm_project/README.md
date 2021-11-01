# zoomcap
midterm project for ML zoomcamp course: https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp

# link to dataset
Medical cost personal dataset from kaggle:
https://www.kaggle.com/mirichoi0218/insurance

# goal of the midterm project
The task of the project is to create a model that can predict the personal insurance costs (column: charges) by using the provided features.

# description of dataset (columns)
- age: age of primary beneficiary
- sex: insurance contractor gender, female, male
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
- children: Number of children covered by health insurance / Number of dependents
- smoker: Smoking
- region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
- charges: Individual medical costs billed by health insurance (target)

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