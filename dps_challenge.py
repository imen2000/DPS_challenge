import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

accidents = pd.read_csv('..\dps_challenge\monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv')

accidents

accidents = accidents.iloc[:,[0,1,2,3,4]]

accidents = accidents.rename(columns= {'MONATSZAHL':'Category','AUSPRAEGUNG':'Accident-type','JAHR':'Year','MONAT':'Month','WERT':'Value'})

accidents

accidents_copy = accidents[accidents['Accident-type']=='insgesamt']
# Group by 'categorie', 'annee' (year), and sum the 'nombre_achat'
category_yearly = accidents_copy.groupby(['Category', accidents_copy['Year']])['Value'].sum().reset_index()

# Plotting
plt.figure(figsize=(12, 8))

# Iterate over unique categories and plot the time series for each
for category in category_yearly['Category'].unique():
    category_data = category_yearly[category_yearly['Category'] == category]
    plt.plot(category_data['Year'], category_data['Value'], label=f'{category}')

    # Ajouter des points à chaque donnée pour chaque catégorie
    plt.scatter(category_data['Year'], category_data['Value'])

plt.xlabel('Year')
plt.ylabel('Total Number of accidents')
plt.title('Yearly number of accidents by Category')
plt.legend()
plt.tight_layout()

# Définir les étiquettes de l'axe des abscisses
plt.xticks(category_yearly['Year'].unique())  # Utilisez les valeurs uniques des années

# Show the plot
plt.show()

accidents= accidents[accidents['Year']<=2020]

accidents['Year'].unique()

print(accidents.isnull().sum()) #Number of empty cells for each column

print(accidents.duplicated().sum()) #verify_duplicate

accidents['Month'] = accidents['Month'].apply(lambda x: str(x)[4:6])

accidents

accidents = pd.get_dummies(accidents,drop_first=True, columns=['Category','Accident-type'])
accidents.head()

accidents.rename(columns={'Accident-type_mit Personenschäden':'Accident-type_mitPersonenschäden' })

accidents.info()

X = accidents.drop("Value", axis=1)
y = accidents["Value"]

random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)
y_pred = random_forest_reg.predict(X)

error = np.sqrt(mean_squared_error(y, y_pred))
print("{:,.02f}".format(error))

data_to_predict = {
    'Category': ['Alkoholunfälle'],
    'Accident-type': ['insgesamt'],
    'Year': [2021],
    'Month': ['01']
}

input_data = pd.DataFrame(data_to_predict)

input_data = pd.get_dummies(input_data, drop_first=True, columns=['Category','Accident-type'])

missing_cols = set(X.columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0

input_data = input_data[X.columns]


predicted_value = random_forest_reg.predict(input_data)
print(predicted_value)

import joblib

data = {"model": random_forest_reg, "training_data": X}


joblib.dump(data, 'saved_steps.joblib')