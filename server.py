from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo XGBoost
with open('modelo_xgboost.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

# Cargar el RobustScaler
with open('robust_scaler_c.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()
    entradas = np.array([
        datos['temperature'],
        datos['pressure'],
        datos['vibration'],
        datos['humidity'],
        datos['equipment_turbine'],
        datos['equipment_pump']
    ]).reshape(1, -1)

    # Estandarizar las entradas con RobustScaler
    entradas_escaladas = scaler.transform(entradas)

    # Realizar la predicción con el modelo XGBoost
    prediccion = modelo.predict(entradas_escaladas)[0]
    return jsonify({'resultado': str(prediccion)})

if __name__ == '__main__':
    app.run(debug=True)