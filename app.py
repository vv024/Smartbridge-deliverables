# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:44:04 2023

@author: HP
"""

import joblib
from flask import Flask, render_template,request

app = Flask(__name__,static_folder='static')

# Load your pre-trained model
model = joblib.load('model.pkl')

# Load your preprocessing transformer
transform = joblib.load('transform.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Get the values of all features from the form submission in index page
    energy = request.args.get('energy')
    loudness = request.args.get('loudness')
    speechiness = request.args.get('speechiness')
    acousticness = request.args.get('acousticness')
    instrumentalness = request.args.get('instrumentalness')
    liveness = request.args.get('liveness')
    valence = request.args.get('valence')
    tempo = request.args.get('tempo')
    duration_ms = request.args.get('duration_ms')
    
    data = [[energy,loudness,speechiness,acousticness,instrumentalness,liveness,
             valence,tempo,duration_ms]]

    # Apply the preprocessing transformer to the data
    transformed_data = transform.transform(data)

    # Use your trained model to predict the genre
    predicted_genre_index =int( model.predict(transformed_data))
    
    # Retrieve the corresponding genre label based on the predicted index
    genre_labels = ['Dark Trap','Emo','Hiphop','Pop','Rap','RnB','Trap Metal',
                    'Underground Rap','dnb','hardstyle','psytrance','techhouse',
                    'techno','trance','trap'] 
    # Replace with your actual genre labels
    predicted_genre = genre_labels[predicted_genre_index]

    return render_template('results.html', genre=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
