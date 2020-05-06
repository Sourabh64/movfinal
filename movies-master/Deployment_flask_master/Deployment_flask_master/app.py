import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import dir_dict, actor1_dict, actor2_dict
from model import dir_adv_dict, actor1_adv_dict, actor2_adv_dict
from model import dir_com_dict, actor1_com_dict, actor2_com_dict
from model import dir_rom_dict, actor1_rom_dict, actor2_rom_dict


app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))
model_adv = pickle.load(open('model2.pkl', 'rb'))
model_com = pickle.load(open('model3.pkl', 'rb'))
model_rom = pickle.load(open('model4.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/act')
def act():
    '''
    For rendering results on HTML GUI
    '''
    return render_template('action.html')

@app.route('/action',methods=['GET', 'POST'])
def action():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method== 'POST':  
        dir_name=request.form.get('director_name')
        dir_score=dir_dict[dir_name]
        actor1_name= request.form.get('actor1_name')
        actor1_score=actor1_dict[actor1_name]
        actor2_name=request.form.get('actor2_name')
        actor2_score=actor2_dict[actor2_name]
    #actor_name1=request.args.get('test_score')
   # actor_score1=actor1_dict.get(actor_name1,"0")       
    final_features=np.array([[dir_score,actor1_score,actor2_score]])
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('action.html', prediction_text='IMDB of the movie is estimated to be {}'.format(output))

@app.route('/adv')
def adv():
    '''
    For rendering results on HTML GUI
    '''
    return render_template('adventure.html')

@app.route('/adventure',methods=['GET', 'POST'])
def adventure():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method== 'POST':  
        dir_name=request.form.get('director_name')
        dir_score=dir_adv_dict[dir_name]
        actor1_name= request.form.get('actor1_name')
        actor1_score=actor1_adv_dict[actor1_name]
        actor2_name=request.form.get('actor2_name')
        actor2_score=actor2_adv_dict[actor2_name]
        
    
    final_features=np.array([[dir_score,actor1_score,actor2_score]])
    prediction = model_adv.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('adventure.html', prediction_text='IMDB of the movie is estimated to be {}'.format(output))

@app.route('/com')
def com():
    '''
    For rendering results on HTML GUI
    '''
    return render_template('comedy.html')

@app.route('/comedy',methods=['GET', 'POST'])
def comedy():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method== 'POST':  
        dir_name=request.form.get('director_name')
        dir_score=dir_com_dict[dir_name]
        actor1_name= request.form.get('actor1_name')
        actor1_score=actor1_com_dict[actor1_name]
        actor2_name=request.form.get('actor2_name')
        actor2_score=actor2_com_dict[actor2_name]
    #actor_name1=request.args.get('test_score')
   # actor_score1=actor1_dict.get(actor_name1,"0")       
    final_features=np.array([[dir_score,actor1_score,actor2_score]])
    prediction = model_com.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('comedy.html', prediction_text='IMDB of the movie is estimated to be {}'.format(output))

@app.route('/rom')
def rom():
    '''
    For rendering results on HTML GUI
    '''
    return render_template('romantic.html')

@app.route('/romantic',methods=['GET', 'POST'])
def romantic():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method== 'POST':  
        dir_name=request.form.get('director_name')
        dir_score=dir_rom_dict[dir_name]
        actor1_name= request.form.get('actor1_name')
        actor1_score=actor1_rom_dict[actor1_name]
        actor2_name=request.form.get('actor2_name')
        actor2_score=actor2_rom_dict[actor2_name]
    #actor_name1=request.args.get('test_score')
   # actor_score1=actor1_dict.get(actor_name1,"0")       
    final_features=np.array([[dir_score,actor1_score,actor2_score]])
    prediction = model_rom.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('romantic.html', prediction_text='IMDB of the movie is estimated to be {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)