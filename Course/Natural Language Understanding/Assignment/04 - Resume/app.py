from flask import Flask, render_template, request, session
from flask_wtf import FlaskForm
from wtforms import SubmitField,StringField, BooleanField, RadioField, SelectField, TextAreaField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'

class MyForm(FlaskForm):
    name = StringField('Insert you name',validators=[DataRequired()])
    isAccept = BooleanField("Accept Information")
    gender = RadioField("Gender",choices=[('Male','Male'),('Female','Female'),('Other','Other')])
    skill = SelectField("Skill",choices=[('English'),('English'),('Sing'),('Sing')])
    submit = SubmitField('Save')
    address = TextAreaField('Your address')
    
@app.route('/', methods = ['GET','POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        session['name'] = form.name.data 
        session['isAccept'] = form.isAccept.data
        session['gender'] = form.gender.data
        session['skill'] = form.skill.data
        session['address'] = form.address.data
        #clear data
        form.name.data = ""
        form.isAccept.data = ""
        form.gender.data = ""
        form.address.data = ""
    return render_template("index.html", form = form)

@app.route('/about')
def about():
    product = ["cloths","iron","napkin","mouse","keyboard"]
    return render_template("about.html",myproduct = product)

@app.route('/admin')
def profile():
    #Name Age
    data = {'name' :'Todsavad','age' : 23,'username' :"guntsv"}
    return render_template("admin.html", data = data)

@app.route('/upload')
def resume_Parser():
    #Name Age
    data = {'name' :'Todsavad','age' : 23,'username' :"guntsv"}
    return render_template("upload.html", data = data)

@app.route('/lab05')
def classification():
    #Name Age
    return render_template("lab05.html")

if __name__ == "__main__":
    app.run(debug=True)