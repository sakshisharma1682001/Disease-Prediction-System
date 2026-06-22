from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/disease_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        fever = int(request.form["fever"])
        cough = int(request.form["cough"])
        headache = int(request.form["headache"])
        fatigue = int(request.form["fatigue"])

        result = model.predict([[fever, cough, headache, fatigue]])

        prediction = result[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)