"""
predict
"""
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':
    model = load_model("../outputs")

    print(model.predict(('it is great', 'it is wonderful', 'lovely', 'not good', 'it is bad')))
