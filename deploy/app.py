"""
predict
"""
from fastapi import FastAPI
from tensorflow.python.keras.models import load_model

model = load_model("./outputs")
app = FastAPI(title="MLOps Basics App")


@app.get("/predict")
async def get_prediction(text: str):
    """
    推理接口
    :param text:
    :return:
    """
    result = model.predict((str(text),))
    return result
