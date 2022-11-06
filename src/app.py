from flask import Flask, render_template, request
from src.predict import predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template(
        "index.html")

@app.route('/', methods=['POST'])
def my_form_post():
    features = [
        request.form['dance'], 
        request.form['cheer'], 
        request.form['acoustic'], 
        10]
    features = [(int(x)/10) for x in features]
    try:
        playlist = predict(request.form['lyrics'], features)
        return render_template(
        "results.html", playlist=playlist)
    except Exception as e:
        message = f'Error: {e}. Please try again!'
        return message
    

if __name__ == "__main__":
    app.run()

