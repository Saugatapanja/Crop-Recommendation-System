from flask import Flask, render_template, request
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model=joblib.load('model.joblib')

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('Crop_Home.html')


@app.route('/predict', methods=['POST'])
def home():
    features=[int(x) for x in request.form.values()]
    arr=[np.array(features)]
    print(arr)
    pred = model.predict(arr)
    print(pred)
    return render_template('Recommended_Crop.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)

    