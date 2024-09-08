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

class PredRequest(BaseModel):
    your: str
    parameters: list[int]

class PredResponse(BaseModel):
    some_number: int
