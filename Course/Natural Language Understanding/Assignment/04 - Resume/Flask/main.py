from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField,StringField, BooleanField, RadioField, SelectField, TextAreaField,FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
from pdf import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        # return "File has been uploaded."
        return redirect(url_for('convert_file', name=file))
    return render_template('upload.html', form=form)

@app.route('/result/<name>')
def convert_file(name,num_page=0):
    skills = readPDF(name,num_page)
    # return send_from_directory(app.config["UPLOAD_FOLDER"], name)
    return render_template("result.html",skills=skills)

@app.route('/lab05')
def classification():
    return render_template("lab05.html")

if __name__ == "__main__":
    app.run(debug=True)