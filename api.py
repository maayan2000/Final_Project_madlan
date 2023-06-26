import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = ['room_number', 'Area', 'hasElevator', 'hasParking', 'hasBars',
                'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad',
                'handicapFriendly', 'floor', 'total_floors',
                'City_נהריה', 'City_שוהם', 'City_אילת', 'City_אריאל', 'City_באר שבע',
                'City_בית שאן', 'City_בת ים', 'City_גבעת שמואל', 'City_דימונה',
                'City_הוד השרון', 'City_הרצליה', 'City_זכרון יעקב', 'City_חולון',
                'City_חיפה', 'City_יהוד מונוסון', 'City_ירושלים', 'City_כפר סבא',
                'City_מודיעין מכבים רעות', 'City_נהריה', 'City_נהרייה',
                'City_נוף הגליל', 'City_נס ציונה', 'City_נתניה', 'City_פתח תקווה',
                'City_צפת', 'City_קרית ביאליק', 'City_ראשון לציון', 'City_רחובות',
                'City_רמת גן', 'City_רעננה', 'City_שוהם', 'City_תל אביב']
    feature_values = [request.form.get(feature) for feature in features]

    # Load the model
    model_path = 'trained_model.pkl'
    if not os.path.isfile(model_path):
        return render_template('index.html', prediction_text='Model file not found.')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Convert the feature values to a 2D array
    feature_array = [list(map(int, feature_values))]

    # Make the prediction
    prediction = model.predict(feature_array)[0]

    output_text = f"The predicted value is: {prediction}"

    return render_template('index.html', prediction_text=output_text)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
