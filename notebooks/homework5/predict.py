import pickle

model_file = 'model1.bin'
vec_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(vec_file, 'rb') as f_in:
    dv = pickle.load(f_in)


def predict():
    input = dv.transform([customer])
    y_pred = model.predict_proba(input)[0, 1]
    return y_pred


if __name__ == '__main__':
    customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}
    y_pred = predict()
    print(f'customer chrun probability: {y_pred}')
