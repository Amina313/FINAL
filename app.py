import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import joblib
data = pd.read_csv(r'C:\Users\User\OneDrive\Рабочий стол\курсы\уроки курсов\FINAL\data.csv')

home_data = data.copy()

home_data['baths'] = home_data['baths'].str.extract('(\d+)')
home_data['baths'] = pd.to_numeric(home_data['baths'])

home_data['beds'] = home_data['beds'].str.extract('(\d+)')
home_data['beds'] = pd.to_numeric(home_data['beds'])

home_data['zipcode'] = home_data['zipcode'].str.extract('(\d+)')
home_data['zipcode'] = pd.to_numeric(home_data['zipcode'])

home_data['stories'] = home_data['stories'].str.extract('(\d+)')
home_data['stories'] = pd.to_numeric(home_data['stories'])

home_data['sqft'] = home_data['sqft'].replace({'sqft': '', ',':''}, regex=True)
home_data['sqft'] = pd.to_numeric(home_data['sqft'], errors='coerce')

home_data['target'] = home_data['target'].replace({'\$': '', ',':''}, regex=True)
home_data['target'] = pd.to_numeric(home_data['target'], errors='coerce')

home_data['MlsId'] = home_data['MlsId'].fillna('')
home_data['mls-id'] = home_data['mls-id'].fillna('')
home_data['MlsId'] = home_data['MlsId'] + home_data['mls-id']
del home_data['mls-id']

home_data['PrivatePool'] = home_data['PrivatePool'].fillna('')
home_data['private pool'] = home_data['private pool'].fillna('')
home_data['PrivatePool'] = home_data['PrivatePool'] + home_data['private pool']
home_data['PrivatePool'] = home_data['PrivatePool'].replace('', 'No')
del home_data['private pool']

labelencoder = LabelEncoder()
home_data['city'] = labelencoder.fit_transform(home_data['city'])
home_data['state'] = labelencoder.fit_transform(home_data['state'])

home_data['status'] = home_data['status'].replace({'for sale': 'For sale', 'Coming sooon: (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}\.': 'Coming soon'}, regex=True)

home_data['PrivatePool'] = home_data['PrivatePool'].replace('yes', 'Yes')

home_data['propertyType'] = home_data['propertyType'].replace({'single-family home': 'Single Family Home', 'Single Family': 'Single Family Home', 'condo': 'Condo', 'Land': 'lot/land'})

home_data['fireplace'] = home_data['fireplace'].replace({'yes': 'Yes', 'Fireplace': 'Yes'})

home_data['status'] = labelencoder.fit_transform(home_data['status'])
home_data['propertyType'] = labelencoder.fit_transform(home_data['propertyType'])
home_data['fireplace'] = labelencoder.fit_transform(home_data['fireplace'])
home_data['PrivatePool'] = labelencoder.fit_transform(home_data['PrivatePool'])

home_data = home_data[['status', 'propertyType', 'baths', 'fireplace', 'city', 'sqft', 'zipcode', 'beds', 'state', 'stories', 'PrivatePool', 'target']]

home_data.fillna(0, inplace=True)

X = home_data[['status', 'propertyType', 'baths', 'fireplace', 'city', 'sqft', 'zipcode', 'beds', 'state', 'stories', 'PrivatePool']]
y = home_data[['target']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

rf_model = RandomForestRegressor( n_estimators = 103,
                                 max_depth = 30, min_samples_leaf = 2)
rf_model.fit(X_train, y_train)


joblib.dump(model, 'model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data, index=[0])
    
    prediction = model.predict(input_data)
    
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)