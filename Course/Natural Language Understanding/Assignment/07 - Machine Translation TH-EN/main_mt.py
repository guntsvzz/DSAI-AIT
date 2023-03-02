from flask import Flask, render_template, request, redirect, url_for, session, flash
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/mt', methods = ['GET','POST'])
def machinetranslation():
    source = request.args.get('source')
    target = request.args.get('targets')
    data = {"source":source,"target":target}
    return render_template("mt.html", data = data)

if __name__ == "__main__":
    app.run(debug=True)