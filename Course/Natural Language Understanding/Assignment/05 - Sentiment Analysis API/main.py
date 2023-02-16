from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from LSTM_predict import PosNeg, prediction,Reddit
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

####Reddit API Part
# reddit crawler
import pandas as pd
import praw
from secretkey import *

reddit = praw.Reddit(client_id=your_client_id,
                     client_secret=your_client_secret,
                     user_agent=your_user_name,
                     check_for_async=False)

class MyForm(FlaskForm):
    name = StringField('Insert your topic', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/sentiment', methods = ['GET','POST'])
def sentiment():
    name = False
    result = False
    pos = False
    neg = False
    plot_url = False
    form = MyForm()
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        form.name.data = ""
        #Reddit Part
        result = Reddit(name,reddit,None)
        #Pos/Neg Part
        pos,neg,plot_url = PosNeg(result)
    return render_template("sentiment.html",form=form,name=name,results=result,pos=pos,neg=neg,plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)