import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import LabelEncoder
import pickle
from madlan_data_prep import prepare_data

def train_and_evaluate_model(data):
    # Omit unnecessary columns
    data = data.drop(columns=['type', 'Street', 'number_in_street', 'city_area',
                             'num_of_images', 'floor_out_of', 'entranceDate ',
                              'condition ', 'furniture ', 'publishedDays ', 'description ', 'Unnamed: 23'])

    # Calculate the average floor and total_floors as integers
    average_floor = data['floor'].mean().astype(float)
    average_total_floors = data['total_floors'].mean().astype(float)

    # Replace missing values with the average
    data['floor'].fillna(average_floor, inplace=True)
    data['total_floors'].fillna(average_total_floors, inplace=True)

    # Convert 'current_entrance_date' column to numeric
    label_encoder = LabelEncoder()
    data['current_entrance_date'] = label_encoder.fit_transform(data['current_entrance_date'].astype(str))

    # Perform one-hot encoding on 'City' column
    data = pd.get_dummies(data, columns=['City'], prefix='City')

    # Split the data into features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Initialize the Elastic Net model
    model = ElasticNetCV(cv=10)

    # Train the model
    model.fit(X, y)

    # Make predictions on the training data
    y_pred = model.predict(X)

    # Calculate the MSE
    mse = mean_squared_error(y, y_pred)
    print("Mean Squared Error (MSE):", mse)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X, y, cv=10)

    return model, scores

# Read the file and preprocess the data
filename = "output_all_students_Train_v10.csv"
preprocessed_data = prepare_data(filename)

# Train and evaluate the model
model, scores = train_and_evaluate_model(preprocessed_data)

# Save the trained model
pickle.dump(model, open("trained_model.pkl", "wb"))
