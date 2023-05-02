from flask import Flask, render_template, redirect, session
from functools import wraps
import pymongo
from mic import list
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRETKEY")

# Database
client = pymongo.MongoClient(os.getenv("MONGOURI"))
db = client.vts

# Decorators
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/')
    return wrap

# Routes
from user import routes

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard/')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/main/')
@login_required
def main():
    return render_template('main.html' , list=list)

@app.route('/record/')
@login_required
def record():
    return render_template('record.html')


