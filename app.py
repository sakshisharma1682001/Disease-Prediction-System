from flask import Flask, render_template, request
import joblib

print("DEBUG: App starting...")

app = Flask(__name__)
print("DEBUG: Flask app created")

model = joblib.load("model/disease_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    print(f"DEBUG: Route hit with method = {request.method}")
    prediction = ""

    if request.method == "POST":
        print("DEBUG: POST request received!")
        fever = int(request.form["fever"])
        cough = int(request.form["cough"])
        headache = int(request.form["headache"])
        fatigue = int(request.form["fatigue"])

        result = model.predict([[fever, cough, headache, fatigue]])

        prediction = result[0]
        print(f"DEBUG: Prediction = {prediction}")

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=False)