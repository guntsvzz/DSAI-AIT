from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from LSTM_predict import *
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
your_client_id='ElyKc6o3du1IMb5LP2HYjg'
your_client_secret='YimfVn3bTLyVFu_XkJuLxrNkR3vHAQ'
your_user_name='guntsv'

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
    results = 0,0
    form = MyForm()
    print(form.validate_on_submit())
    if form.validate_on_submit():
        name = form.name.data 
        form.name.data = ""
        subreddit = reddit.subreddit(name)
        topics = [*subreddit.top(limit=50)] # top posts all time
        # print(len(topics))
        title = [n.title for n in topics]
        df_topics = pd.DataFrame({"title": title})
        results = prediction(df_topics['title'])
    return render_template("sentiment.html",form=form,name=name,results=results)

if __name__ == "__main__":
    app.run(debug=True)