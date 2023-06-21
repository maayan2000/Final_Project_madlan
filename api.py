import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
rf_model = pickle.load(open('trained_model.pkl', 'rb'))

# City mapping dictionary
city_mapping = {
    'נהריה': 0.25806451612903225,
    'שוהם': 0.8387096774193549,
    'אילת': 0.1935483870967742,
    'אריאל': 0.22580645161290322,
    'באר שבע': 0.06451612903225806,
    'בית שאן': 0.0967741935483871,
    'בת ים': 0.5483870967741935,
    'גבעת שמואל': 0.6774193548387096,
    'דימונה': 0.0,
    'הוד השרון': 0.9354838709677419,
    'הרצליה': 0.7419354838709677,
    'זכרון יעקב': 0.7096774193548387,
    'חולון': 0.41935483870967744,
    'חיפה': 0.2903225806451613,
    'יהוד מונוסון': 0.6129032258064516,
    'ירושלים': 1.0,
    'כפר סבא': 0.5806451612903226,
    'מודיעין מכבים רעות': 0.6451612903225806,
    'נהריה': 0.16129032258064516,
    'נהרייה': 0.45161290322580644,
    'נוף הגליל': 0.03225806451612903,
    'נס ציונה': 0.7741935483870968,
    'נתניה': 0.5161290322580645,
    'פתח תקווה': 0.4838709677419355,
    'צפת': 0.3548387096774194,
    'קרית ביאליק': 0.12903225806451613,
    'ראשון לציון': 0.3870967741935484,
    'רחובות': 0.3225806451612903,
    'רמת גן': 0.8709677419354839,
    'רעננה': 0.8064516129032258,
    'שוהם': 0.967741935483871,
    'תל אביב': 0.9032258064516129
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [
        'City', 'room_number', 'Area', 'city_area', 'hasElevator', 'hasParking',
        'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad',
        'handicapFriendly', 'floor', 'total_floors', 'current_entrance_date'
    ]
    
    feature_values = [request.form.get(feature) for feature in features]
    
    # Map the City feature to its corresponding numerical value
    city_value = city_mapping.get(feature_values[0])
    feature_values[0] = city_value if city_value is not None else 0.0
    
    #final_features = np.array(feature_values).reshape(1, -1)
    # Ensure that all values in final_features are valid and convertable to float
    final_features = [float(value) if value.isdigit() else 0.0 for value in features]

    # Make the prediction
    prediction = rf_model.predict([final_features])[0]
    
    output_text = f"The predicted price is: {prediction}"
    
    return render_template('index.html', prediction_text=output_text)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
