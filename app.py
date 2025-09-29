
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json(force=True)
        df_predict = pd.DataFrame([data])

        # Ensure the columns are in the same order as the training data
        # This is a crucial step for correct predictions
        # We need the original columns from X, excluding 'Placement'
        # Assuming X_train columns are representative of the required input features
        required_columns = X_train.columns.tolist()
        df_predict = df_predict[required_columns]

        # Preprocess the 'Internship_Experience' column if it exists
        if 'Internship_Experience' in df_predict.columns:
             df_predict['Internship_Experience'] = df_predict['Internship_Experience'].apply(lambda x: 1 if x == 'Yes' else 0)


        # Make prediction
        prediction = model.predict(df_predict)

        # Convert prediction to a readable format (e.g., 'Yes' or 'No' for Placement)
        # Assuming the label encoder was used and 0=No, 1=Yes
        prediction_label = 'Yes' if prediction[0] == 1 else 'No'


        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # To run locally, you might use app.run(debug=True)
    # For deployment, consider using a production-ready WSGI server like Gunicorn or uWSGI
    # In Colab, you can use ngrok to expose your local server
    app.run(host='0.0.0.0', port=5000)
