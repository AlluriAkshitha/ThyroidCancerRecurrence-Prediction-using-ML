from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('thyroid_cancer_rf_model.pkl')  # Make sure this file is in your root project folder!

# Define the order of features expected by the model
feature_names = ['Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
                 'Thyroid Function', 'Physical Examination', 'Adenopathy',
                 'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

# Preprocess input data from form
def preprocess_input(data):
    encoding_maps = {
        'Gender': {'male': 1, 'female': 0},
        'Smoking': {'yes': 1, 'no': 0},
        'Hx Smoking': {'yes': 1, 'no': 0},
        'Hx Radiothreapy': {'yes': 1, 'no': 0},
        'Thyroid Function': {'normal': 0, 'hyper': 1, 'hypo': 2},
        'Physical Examination': {'normal': 0, 'abnormal': 1},
        'Adenopathy': {'yes': 1, 'no': 0},
        'Pathology': {'benign': 0, 'malignant': 1},
        'Focality': {'unifocal': 0, 'multifocal': 1},
        'Risk': {'low': 0, 'intermediate': 1, 'high': 2},
        'Response': {'excellent': 0, 'indeterminate': 1, 'biochemical incomplete': 2, 'structural incomplete': 3},
        'T': {'t0': 0, 't1': 1, 't2': 2, 't3': 3, 't4': 4},
        'N': {'n0': 0, 'n1': 1, 'n2': 2, 'n3': 3},
        'M': {'m0': 0, 'm1': 1},
        'Stage': {'stage i': 0, 'stage ii': 1, 'stage iii': 2, 'stage iv': 3,
                  'i': 0, 'ii': 1, 'iii': 2, 'iv': 3}
    }

    for key in data:
        value = data[key]

        if value is None:
            data[key] = 0  # Default fallback for missing fields
            continue

        if key in encoding_maps:
            value = value.lower().strip()
            data[key] = encoding_maps[key].get(value, 0)  # Default to 0 if not found
        else:
            try:
                data[key] = float(value)
            except:
                data[key] = 0  # Another fallback for invalid numeric input

    return data

@app.route('/')
def home():
    return render_template('index.html')  # 👈 Make sure you have templates/index.html

@app.route('/predict', methods=['POST'])
def predict():
    form_data = {feat: request.form.get(feat) for feat in feature_names}
    processed_data = preprocess_input(form_data)
    input_df = pd.DataFrame([processed_data], columns=feature_names)

    prediction = model.predict(input_df)[0]

    result = (
        "🟢 Everything looks good! No signs of cancer coming back."
        if prediction == '0'
        else "🔴 There may be a chance of cancer returning. Please consult a doctor for further advice."
    )

    return render_template('result.html', prediction=result)  # 👈 Make sure templates/result.html exists!

if __name__ == '__main__':
    app.run(debug=True)
