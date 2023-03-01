from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import pandas as pd

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
    source = StringField('Type something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/mt', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()

    source = False
    print(form.validate_on_submit())
    if form.validate_on_submit():
        source = form.source.data 
        target = 5 #change later
        form.source.data = ""
    return render_template("mt.html", form=form, source=source, target=target)

if __name__ == "__main__":
    app.run(debug=True)