import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('KAG_energydata_complete.csv')


X = df.drop(columns=['date', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'T9', 'RH_9', 
                     'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2'], axis=1)
y = df['Appliances']  # Bağımlı değişken (hedef)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
idge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)


y_pred = ridge_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


new_data = pd.DataFrame({
    'date':[60], 
    'lights':[60],
    'T1':[60], 
    'RH_1':[60], 
    'T2':[60], 
    'RH_2':[60], 
    'T3':[60],
    'RH_3':[60],
    'T4':[60], 
    'T9':[60], 
    'RH_9':[60], 
    'T_out':[60], 
    'Press_mm_hg':[60], 
    'RH_out':[60], 
    'Windspeed':[60],
    'Visibility':[60], 
    'Tdewpoint':[60], 
    'rv1':[60], 
    'rv2':[60]
})


columns = X_train.columns


new_data = new_data.reindex(columns=columns, fill_value=0)


prediction = ridge_model.predict(new_data)
print(f'guess: {prediction}')
