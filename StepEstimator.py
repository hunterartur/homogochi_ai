from pydantic import BaseModel
class StepEstimator(BaseModel):
    '''
    Class, describing the variables in the input
    '''
    gender: float
    age: float
    height: float
    weight: float
    daily_activity: float

