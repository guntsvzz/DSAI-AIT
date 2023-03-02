from flask import Flask, render_template, request, redirect, url_for, session, flash
from wtforms.validators import DataRequired
from TestingNMT import *
# import TestingNMT
import torch

import torch, torchdata, torchtext
from torch import nn
import torch.nn.functional as F

import random, math, time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#make our work comparable if restarted the kernel
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

variants = 'additive'
save_path = './models/Seq2SeqPackedAttention_additive.pt'

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
    if request.method == 'POST':
        source = request.form.get('source')
        predict = translation(source, variants, save_path, device)
    else:
        source = ' '
        predict = ' '
    # predict = 555
    data = {"source":source, "predict":predict}
    return render_template("mt.html", data = data)

if __name__ == "__main__":
    app.run(debug=True)