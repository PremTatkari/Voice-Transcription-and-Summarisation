from flask import Flask, jsonify, request, session, redirect
from passlib.hash import pbkdf2_sha256
import uuid
from app import db

class User:

    def start_session(self, user):
        del user['password']
        session['logged_in'] = True
        session['user'] = user
        return jsonify(user), 200

    def signup(self):

        user = {
            "_id": uuid.uuid4().hex,
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "password": request.form.get('password')
        }
     
        # Password encryption
        user['password'] = pbkdf2_sha256.encrypt(user['password'])

        # Check for existing users
        if db.users.find_one({ "email": user['email'] }): 
            return jsonify({ "error": "Email address already in use" }), 400


        if db.users.insert_one(user):
            return self.start_session(user)

        return jsonify({ "error": "Signup failed" }), 400
    
    def login(self):
        user = db.users.find_one({
            "email": request.form.get('email')
        })

        if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
            return self.start_session(user)
        
        return jsonify({ "error": "Invalid login credentials" }), 401
    
    def logout(self):
        session.clear()
        return redirect('/')
    

# There are many techniques available to generate extractive summarization
# to keep it simple, I will be using an unsupervised learning approach to
# find the sentences similarity and rank them. Summarization can be
# defined as a task of producing a concise and fluent summary while
# preserving key information and overall meaning. One benefit of this will
# be, you don’t need to train and build a model prior start using it for
# your project. It’s good to understand Cosine similarity to make the best
# use of the code you are going to see. Cosine similarity is a measure of
# similarity between two non-zero vectors of an inner product space that
# measures the cosine of the angle between them. Its measures cosine of
# the angle between vectors. The angle will be 0 if sentences are similar.