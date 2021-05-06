import flask
import pickle
import numpy 
import pandas as pd
import pygad
import pygad.nn
import pygad.gann


fn = 'C:/Users/hitat/anaconda3/PROJECT/model/spotify_data.pkl'
model_instance = pickle.load(open(fn,'rb'))

app = flask.Flask(__name__, template_folder='C:/Users/hitat/anaconda3/PROJECT')
@app.route('/', methods=['GET','POST'])

def main():
    if flask.request.method == 'GET':
        return (flask.render_template('audioo.html'))

    if flask.request.method == 'POST': 
        valence = flask.request.form['valence']
        duration_ms = flask.request.form['duration_ms']
        explicit = flask.request.form['explicit']
        danceability = flask.request.form['danceability']
        energy = flask.request.form['energy']
        key = flask.request.form['key']
        mode = flask.request.form['mode']
        speechiness = flask.request.form['speechiness']
        acousticness = flask.request.form['acousticness']
        liveness = flask.request.form['liveness']
        tempo = flask.request.form['tempo']
        time_signature = flask.request.form['time_signature']
        
        a_inputs = numpy.array([[valence, duration_ms, explicit, danceability, energy, key, mode, speechiness, acousticness, 
                                 liveness, tempo, time_signature]])
        a_inputs = a_inputs.astype(float)
        
        prediction = pygad.nn.predict(last_layer=model_instance, data_inputs=a_inputs)
        prediction = str(prediction).strip('[]')
        
        pred = ''
        if int(prediction) == 0:
            pred = 'Flop'
        else: 
            pred = 'Bop'
        
        return (flask.render_template('audioo.html', original_input= {}, result=str(pred)))
        

if __name__ == '__main__':
    app.run()
    
    