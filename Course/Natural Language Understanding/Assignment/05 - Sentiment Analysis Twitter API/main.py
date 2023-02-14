from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, BooleanField, RadioField, SelectField, TextAreaField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

class MyForm(FlaskForm):
    name = StringField('Insert your topic',validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/sentiment')
def sentiment():
    form = MyForm()
    return render_template("sentiment.html",form=form)

if __name__ == "__main__":
    app.run(debug=True)