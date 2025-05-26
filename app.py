from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and the expected column order
model = joblib.load('thyroid_recurrence_rf.pkl')
model_columns = joblib.load('model_columns.pkl')  # List of columns used during model training

# List of features expected from the form
feature_names = [
    'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
    'Thyroid Function', 'Physical Examination', 'Adenopathy',
    'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
]

# Preprocess input data from form
def preprocess_input(data):
    encoding_maps = {
        'Risk': {'Low': 0, 'Intermediate': 1, 'High': 2},
        'Response': {'Excellent': 0, 'Indeterminate': 1, 'Biochemical Incomplete': 2, 'Structural Incomplete': 3},
        'T': {'T1a': 0, 'T1b': 1, 'T2': 2, 'T3a': 3, 'T3b': 4, 'T4a': 5, 'T4b': 6},
        'N': {'N0': 0, 'N1a': 1, 'N1b': 2},
        'M': {'M0': 0, 'M1': 1},
        'Stage': {'I': 0, 'II': 1, 'III': 2, 'IVA': 3, 'IVB': 4},
        'Pathology': {'Papillary': 0, 'Micropapillary': 1, 'Follicular': 2, 'Hurthel cell': 3},
        'Focality': {'Uni-Focal': 0, 'Multi-Focal': 1},
        'Adenopathy': {'No': 0, 'Right': 1, 'Left': 2, 'Bilateral': 3, 'Extensive': 4, 'Posterior': 5},
        'Smoking': {'No': 0, 'Yes': 1},
        'Hx Smoking': {'No': 0, 'Yes': 1},
        'Hx Radiothreapy': {'No': 0, 'Yes': 1},
        'Thyroid Function': {
            'Euthyroid': 0,
            'Clinical Hyperthyroidism': 1,
            'Clinical Hypothyroidism': 2,
            'Subclinical Hyperthyroidism': 3,
            'Subclinical Hypothyroidism': 4
        },
        'Physical Examination': {
            'Multinodular goiter': 0,
            'Single nodular goiter-right': 1,
            'Single nodular goiter-left': 2,
            'Normal': 3,
            'Diffuse goiter': 4
        },
        'Gender': {'F': 0, 'M': 1}
    }

    processed = {}
    for key in data:
        value = data[key]
        if value is None or value == '':
            processed[key] = 0  # Default fallback
            continue
        if key in encoding_maps:
            processed[key] = encoding_maps[key].get(value, 0)  # No .lower().strip()!
        else:
            try:
                processed[key] = float(value)
            except:
                processed[key] = 0
    return processed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    form_data = {feat: request.form.get(feat) for feat in feature_names}
    processed_data = preprocess_input(form_data)
    input_df = pd.DataFrame([processed_data])

    print("Form data:", form_data)
    print("Processed data:", processed_data)
    print("Input DataFrame:", input_df)

    # Ensure input_df matches the model's expected columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Make sure prediction is an integer (0 or 1)
    try:
        prediction = int(prediction)
    except:
        # If prediction is a string, map it
        prediction = 1 if str(prediction).lower() in ['yes', '1'] else 0

    if prediction == 0:
        result = " Everything looks good! No signs of cancer coming back."
    else:
        result = " There may be a chance of cancer returning. Please consult a doctor for further advice."

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
