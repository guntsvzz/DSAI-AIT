from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import pandas as pd
from predict_auto import predict
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
    name = StringField('Type something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/autocomplete', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()
    code = False
    name = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        code = predict(prompt = name, temperature=0.5)
        form.name.data = ""
    return render_template("autocomplete.html",form=form,name =name, code=code)

if __name__ == "__main__":
    app.run(debug=True)