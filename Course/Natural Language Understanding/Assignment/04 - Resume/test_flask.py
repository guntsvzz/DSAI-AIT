from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    data = {"name":"Todsavad","age":30,"salary":10000}
    return render_template("index.html",mydata = data)

@app.route('/about')
def about():
    product = ["cloths","iron","napkin","mouse","keyboard"]
    return render_template("about.html",myproduct = product)

@app.route('/admin')
def profile():
    #Name Age
    data = {'name' :'Todsavad','age' : 23,'username' :"guntsv"}
    return render_template("admin.html", data = data)

@app.route('/sendData')
def signupForm():
    fname = request.args.get('fname')
    description = request.args.get('description')
    return render_template("thankyou.html", data = {"name":fname,"description":description})

if __name__ == "__main__":
    app.run(debug=True)