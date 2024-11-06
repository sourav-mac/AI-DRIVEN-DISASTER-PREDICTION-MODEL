from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

app = Flask(__name__)

# Load models, scaler, and label encoder
earthquake_clf = joblib.load('random_forest_classifier.pkl')
earthquake_reg = joblib.load('random_forest_regressor.pkl')
earthquake_scaler = joblib.load('scaler.pkl')
earthquake_le = joblib.load('label_encoder.pkl')

# Load earthquake data and train model (if not pre-saved)
df = pd.read_csv('q.csv')
df['Origin Time'] = pd.to_datetime(df['Origin Time'].str.replace(' IST', ''), format='%Y-%m-%d %H:%M:%S')
df['Origin Time'] = df['Origin Time'].astype('int64') // 10**9
df['Magnitude'] = df['Magnitude'].str.extract(r'([0-9]+\.[0-9]+)').astype(float)
df = df.dropna()

# Normalize earthquake features
earthquake_scaler = MinMaxScaler()
df[['Lat', 'Long', 'Depth', 'Origin Time']] = earthquake_scaler.fit_transform(df[['Lat', 'Long', 'Depth', 'Origin Time']])

# Train earthquake models
# (Place any new earthquake model code here)

# Load and prepare flood data
flood_data = pd.read_csv('run.csv')
flood_cols = ['Latitude', 'Longitude', 'Rainfall', 'Temperature', 'Humidity', 'River Discharge', 'Water Level', 'Elevation', 'Historical Floods']
flood_data = flood_data[flood_cols]

# Split and scale flood data
X_flood = flood_data.drop(['Historical Floods'], axis=1)
y_flood = flood_data['Historical Floods']
X_flood_train, X_flood_test, y_flood_train, y_flood_test = train_test_split(X_flood, y_flood, test_size=0.25, random_state=42)
flood_scaler = StandardScaler()
X_flood_train_scaled = flood_scaler.fit_transform(X_flood_train)
X_flood_test_scaled = flood_scaler.transform(X_flood_test)

# Train flood model
flood_clf = RandomForestClassifier(n_estimators=100, random_state=42)
flood_clf.fit(X_flood_train_scaled, y_flood_train)

# Save the flood model and scaler
joblib.dump(flood_clf, 'flood_classifier.pkl')
joblib.dump(flood_scaler, 'flood_scaler.pkl')

# Load flood model and scaler
flood_clf = joblib.load('flood_classifier.pkl')
flood_scaler = joblib.load('flood_scaler.pkl')

# Prediction API Routes
@app.route('/')
def index():
    return render_template('index.html')

# Earthquake Prediction Route
@app.route('/predict', methods=['POST', 'GET'])
def predict_earthquake():
    if request.method == 'POST':
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        depth = float(data['depth'])
        input_data = pd.DataFrame({'Lat': [latitude], 'Long': [longitude], 'Depth': [depth], 'Origin Time': [0]})
        input_data[['Lat', 'Long', 'Depth', 'Origin Time']] = earthquake_scaler.transform(input_data[['Lat', 'Long', 'Depth', 'Origin Time']])

        predicted_category_encoded = earthquake_clf.predict(input_data)
        predicted_category = earthquake_le.inverse_transform(predicted_category_encoded)
        predicted_magnitude = earthquake_reg.predict(input_data)

        return jsonify({
            'predicted_category': predicted_category[0],
            'predicted_magnitude': round(predicted_magnitude[0], 2)
        })
    return render_template('earthquake_index.html')

# Flood Prediction Route
@app.route('/predict_flood', methods=['POST', 'GET'])
def predict_flood():
    if request.method == 'POST':
        data = request.json
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        rainfall = float(data['rainfall'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        river_discharge = float(data['river_discharge'])
        water_level = float(data['water_level'])
        elevation = float(data['elevation'])

        # Prepare input data for flood prediction
        input_data = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Rainfall': [rainfall],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'River Discharge': [river_discharge],
            'Water Level': [water_level],
            'Elevation': [elevation]
        })

        # Scale input data
        input_data_scaled = flood_scaler.transform(input_data)

        # Predict flood risk
        flood_prediction = flood_clf.predict(input_data_scaled)
        flood_result = "Flood likely" if flood_prediction[0] == 1 else "Flood unlikely"

        return jsonify({'flood_result': flood_result})
    return render_template('flood_index.html')

# Weather Route
@app.route('/weather', methods=['GET'])
def weather():
    return render_template('weather.html')

if __name__ == '__main__':
    app.run(debug=True)
