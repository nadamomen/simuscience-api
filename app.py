from flask import Flask, request, jsonify
import ai_pipeline

app = Flask(__name__)

ai_pipeline.load_assets()

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    reactants = data.get("reactants")
    conditions = data.get("conditions")

    if reactants is None or conditions is None:
        return jsonify({"error": "Both 'reactants' and 'conditions' are required"}), 400

    X = ai_pipeline.preprocess_input(reactants, conditions)
    y_pred = ai_pipeline.model.predict(X)
    result = ai_pipeline.postprocess_output(y_pred)

    return jsonify({"success": True, "result": result})

if __name__ == "__main__":
    app.run()