import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
encoders_x = None
encoders_y = None
config = None
max_reactants = None

def load_assets():
   
    global model, encoders_x, encoders_y, config, max_reactants
    try:
        model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
        encoders_x = joblib.load(os.path.join(BASE_DIR, "encoders_x.joblib"))
        encoders_y = joblib.load(os.path.join(BASE_DIR, "encoders_y.joblib"))
        config = joblib.load(os.path.join(BASE_DIR, "config.joblib"))
        max_reactants = config["max_reactants"]
        print("Assets loaded successfully.")
    except Exception as e:
        print("Error loading assets:", e)
        raise e

def preprocess_input(reactants_text, conditions_text):
   
    reactants = [r.strip() for r in reactants_text.split("+")]
    encoded = []

    reactant_cols = config["reactant_cols"]

    for i in range(max_reactants):
        col = reactant_cols[i]
        encoder = encoders_x[col]

        if i < len(reactants):
            r = reactants[i]
            if r in encoder.classes_:
                encoded.append(encoder.transform([r])[0])
            else:
                encoded.append(-1)
        else:
            encoded.append(-1)

    cond_encoder = encoders_x["conditions"]
    if conditions_text in cond_encoder.classes_:
        encoded.append(cond_encoder.transform([conditions_text])[0])
    else:
        encoded.append(-1)

    columns = [f"reactant_{i+1}" for i in range(max_reactants)] + ["conditions"]
    return pd.DataFrame([encoded], columns=columns)

def postprocess_output(prediction):
   
    product_encoder = encoders_y["products"]
    safety_encoder = encoders_y["safety"]

    product = product_encoder.inverse_transform([prediction[0][0]])[0]
    safety = safety_encoder.inverse_transform([prediction[0][1]])[0]

    return {
        "product": product,
        "safety": safety
    }