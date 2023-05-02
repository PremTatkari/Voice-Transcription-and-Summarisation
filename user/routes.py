from flask import Flask, redirect
from app import app
from user.models import User
from threading import Thread
from mic import transcript, sumarizer


@app.route('/user/signup', methods=['POST'])
def signup():
    return User().signup()

@app.route('/user/login', methods=['POST'])
def login():
    return User().login()

@app.route('/user/logout')
def logout():
    return User().logout()

@app.route('/main/transcript', methods=('GET','POST'))
def transcription():
    thread = Thread(target=transcript)
    thread.start()
    return ('', 204)

@app.route('/main/transcriptstop', methods=('GET','POST'))
def transcriptionstop():
    global go
    go = False
    print("ABC")
    sumarizer()
    return redirect('/main/')
