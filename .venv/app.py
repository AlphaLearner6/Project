from flask import (Flask, render_template, session)

app = Flask(__name__, template_folder="templates")

app.secret_key = "super secret key"

@app.route('/')
def home():
    session.clear()
    return render_template("index.html")

