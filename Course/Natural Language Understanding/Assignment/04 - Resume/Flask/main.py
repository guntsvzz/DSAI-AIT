from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField,StringField, BooleanField, RadioField, SelectField, TextAreaField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import os
from pdf import *

UPLOAD_FOLDER = '../Flask/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000   

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from flask import send_from_directory

@app.route('/uploads/<name>')
def download_file(name,num_page):
    skills, educations = readPDF(name,num_page)
    # return send_from_directory(app.config["UPLOAD_FOLDER"], name)
    return render_template("result.html",skills=skills,educations=educations)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename, num_page=0))

    return render_template("upload.html")

@app.route('/lab05')
def classification():
    return render_template("lab05.html")

if __name__ == "__main__":
    app.run(debug=True)