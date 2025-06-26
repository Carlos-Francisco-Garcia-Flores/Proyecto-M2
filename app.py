from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo y el scaler
model = joblib.load('modelo_random_forest_T6.pkl')
scaler = joblib.load('scaler_t6.pkl')
app.logger.debug("Model and scaler loaded successfully.")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        data = {
            'mental_health_rating': float(request.form['mental_health_rating']),
            'social_media_hours': float(request.form['social_media_hours']),
            'netflix_hours': float(request.form['netflix_hours']),
            'exercise_frequency': float(request.form['exercise_frequency']),
            'study_hours_per_day': float(request.form['study_hours_per_day']),
            'sleep_hours': float(request.form['sleep_hours'])
        }
        
        app.logger.debug(f"Datos recibidos: {data}")

        # Convertir los datos a un DataFrame
        data_df = pd.DataFrame([data])
        app.logger.debug(f"DataFrame creado:\n{data_df}")

        # Escalar los datos (usando el mismo scaler que en el entrenamiento)
        scaled_data = scaler.transform(data_df)
        app.logger.debug(f"Datos escalados:\n{scaled_data}")

        # Realizar la predicción
        prediction = model.predict(scaled_data)
        app.logger.debug(f"Predicción: {prediction[0]}")

        # Devolver la predicción como JSON
        return jsonify({
            'predicted_score': float(prediction[0]),
            'status': 'success'
        })
        
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)