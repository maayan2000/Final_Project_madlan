#!/usr/bin/env python
# coding: utf-8

# # <h1 align="center"><span style="color:Black; font-family: Tahoma">Final Task</span></h1>
# ## <h1 align="center"><span style="color:Black; font-family: Tahoma">Maayan Elkarif - 211695234</span></h1>
# ## <h1 align="center"><span style="color:Black; font-family: Tahoma">Ofir Tzafrir - 206666950</span></h1>
# ___

# <div>
# <img src="https://media3.giphy.com/media/e8ik35i8LaO3BqRwY6/giphy.gif" width="500"/>
# </div>

# ___

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime

def prepare_data(filename):
    data = pd.read_csv(filename)

    missing_price = data['price'].isnull()
    data = data[~missing_price]
    data = data.reset_index(drop=True)

    # -----------------------------------------------------

    data['price'] = data['price'].apply(lambda x: re.sub(r'[^\d]+', '', str(x)))
    data['price'] = data['price'].replace('', '0')
    data['price'] = data['price'].astype(int)

    data['Area'] = data['Area'].apply(lambda x: re.sub(r'[^\d]+', '', str(x)))
    data['Area'] = data['Area'].replace('', '0')
    data['Area'] = data['Area'].astype(int)

    data['Street'] = data['Street'].str.replace(r'\[|\]', '', regex=True)
    data['Street'] = data['Street'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))

    data['room_number'] = data['room_number'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))
    data['room_number'] = data['room_number'].replace('', '0')

    # -----------------------------------------------------

    data['description '] = data['description '].str.replace(',', '')
    data['city_area'] = data['city_area'].str.replace("'", "").str.replace('"', '')

    # -----------------------------------------------------

    floor_list = []
    total_floor_list = []

    for i in data['floor_out_of']:
        if i == '' or i == 'None' or i == 'NaN':
            i = 'None'

        i = str(i)

        if i != 'None':
            if 'קרקע' in i:
                floor_list.append('0')
            elif 'מרתף' in i:
                floor_list.append('-1')
            else:
                match = re.search(r'קומה (\d+)', i)
                if match:
                    floor_list.append(match.group(1))
                else:
                    floor_list.append('None')

            if len(i) == 2:
                total_floor_list.append('None')
            else:
                match_total = re.search(r'מתוך (\d+)', i)
                if match_total:
                    total_floor_list.append(match_total.group(1))
                else:
                    total_floor_list.append('None')
        else:
            floor_list.append('None')
            total_floor_list.append('None')

    data['floor'] = floor_list
    data['total_floors'] = total_floor_list

    # -----------------------------------------------------

    columns_to_process = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ',
                          'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']

    for column in columns_to_process:
        for i in range(len(data)):
            value = data.loc[i, column]
            if pd.isnull(value):
                data.loc[i, column] = None
            elif column == 'handicapFriendly ' and value == 'לא נגיש לנכים':
                data.loc[i, column] = 0
            elif any(word in str(value).lower() for word in ['יש', 'yes', 'כן', 'true', '1', 'נגיש לנכים']):
                data.loc[i, column] = 1
            else:
                data.loc[i, column] = 0

    for column in columns_to_process:
        data[column] = data[column].replace({'True': 1, 'False': 0, None: 0, '': 0}).astype(int)

    # -----------------------------------------------------

    def map_entrance_date(value):
        if pd.isnull(value) or value == 'לא צויין':
            return 'not_defined'
        elif value == 'גמיש' or value == 'מיידי':
            return 'flexible'
        else:
            try:
                date = datetime.strptime(value, '%d/%m/%Y')
            except:
                return 'not_defined'

            year = date.year
            month = date.month
            day = date.day

            if year == datetime.now().year:
                return 'less_than_6_months'
            elif year == datetime.now().year + 1:
                return 'months_6_12'
            elif year > datetime.now().year + 1:
                return 'above_year'
            else:
                return 'not_defined'

    data['current_entrance_date'] = data['entranceDate '].apply(map_entrance_date).astype('category')

    data['room_number'] = data['room_number'].apply(lambda x: float(x) if x != 'None' else np.nan)
    data['Area'] = data['Area'].apply(lambda x: float(x) if x != 'None' else np.nan)
    data['price'] = data['price'].apply(lambda x: float(x) if x != 'None' else np.nan)
    data['floor'] = data['floor'].apply(lambda x: float(x) if x != 'None' else np.nan)
    data['total_floors'] = data['total_floors'].apply(lambda x: float(x) if x != 'None' else np.nan)

    data['room_number'] = data['room_number'].astype(float)
    data['Area'] = data['Area'].astype(int)
    data['price'] = data['price'].astype(int)
    data['floor'] = data['floor'].astype(float)
    data['total_floors'] = data['total_floors'].astype(float)

    return data
data = prepare_data('output_all_students_Train_v10.csv')
data.head()


# In[30]:


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


# In[31]:


# # Calculate the average price for each city
# city_avg_price = preprocessed_data.groupby('City')['price'].mean()

# # Calculate the normalized values based on the average price
# normalized_values = city_avg_price / preprocessed_data['price'].mean()

# # Create a dictionary to store the city values
# city_values = dict(zip(city_avg_price.index, normalized_values))

# # Print the city values
# for city, value in city_values.items():
#     print(f"City: {city}, Normalized Value: {value}")


# In[ ]:




