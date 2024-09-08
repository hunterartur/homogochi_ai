import uvicorn
from fastapi import FastAPI
from StepEstimator import StepEstimator
import pickle

# Creating the app object and loading the pickled model
app = FastAPI()
pickle_in = open("xgb.pkl","rb")
xgb_regressor = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, Homogochi user'}

# Routing with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To Homogochi App': f'{name}'}


# Exposing the prediction functionality, making a prediction from the passed
#    JSON data and return the predicted number of steps
@app.post('/predict')
def predict_steps(data: StepEstimator):
    print(data)
    data = data.dict()
    # data = data.dict()
    gender=data['gender']
    age=data['age']
    height=data['height']
    weight = data['weight']
    daily_activity=data['daily_activity']

    prediction = xgb_regressor.predict([[gender, age, height, weight, daily_activity]])
    print(prediction)

    return {
        'baseline_steps': prediction
    }

# 5. Run the API with uvicorn
#    by default runs on http://127.0.0.1:8000, we swapped to port 8090
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8090)
    
#uvicorn app:app --reload