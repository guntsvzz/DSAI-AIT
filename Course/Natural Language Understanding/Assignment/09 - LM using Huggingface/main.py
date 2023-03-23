from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import pandas as pd

import torch
from transformers import pipeline

checkpoints = "guntsv/alice-in-ait-accelerate"
pipe = pipeline("text-generation", max_length=30, pad_token_id=0, model=checkpoints) #eos_token_id=0

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

class MyForm(FlaskForm):
    text = StringField('Type something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/storygenerator', methods = ['GET','POST'])
def storygenerator():
    form = MyForm()
    gen = False
    text = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        text = form.text.data 
        gen = pipe(text)[0]["generated_text"]
        print(gen)
        form.text.data = ""
    return render_template("storygenerator.html", form = form, text = text, gen = gen)

if __name__ == "__main__":
    app.run(debug=True)