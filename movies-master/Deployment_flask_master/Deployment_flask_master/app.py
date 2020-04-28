import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import dir_dict,actor1_dict,actor2_dict


app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method == 'POST': 
        dir_name=request.form.get('experience')
        dir_score=dir_dict[dir_name]
        actor1_name= request.form.get('test_score')
        actor1_score=actor1_dict[actor1_name]
        actor2_name=request.form.get('interview_score')
        actor2_score=actor2_dict[actor2_name]
    #actor_name1=request.args.get('test_score')
   # actor_score1=actor1_dict.get(actor_name1,"0")
   #actor_name2=request.args.get('interview_score')
    #actor_score2=actor2_dict.get(actor_name2,"0")
    
   # int_features = [float(x) for x in request.form.values()]
   # final_features = [np.array(int_features)]
        final_features=np.array([[dir_score,actor1_score,actor2_score]])
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

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