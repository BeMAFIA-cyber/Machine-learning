import pandas as pd 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (PolynomialFeatures ,   StandardScaler)
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
Table = pd.read_csv('house.csv', dtype={'Area': 'string'})
Table['Area'] = pd.to_numeric(
    Table['Area']
    .str.strip()
    .str.replace(',', '', regex=False),
    errors='coerce'
)
Table = Table.dropna(subset=['Area', 'Price','Area']).copy()
Table = Table[Table['Area'] <= 1000].copy()
x = Table.drop(columns=['Price', 'Price(USD)'])
y = Table['Price']
y2 = Table['Price(USD)']
for col in ['Parking', 'Elevator', 'Warehouse']:
    x[col] = x[col].astype(int)
#bad_indices = Table.index[Table["Area"] > 1000]
x_train , x_test ,y_train , y_test,y2_train,y2_test = train_test_split(x,y,y2 , test_size=0.2 , random_state=42)
numeric_features = x_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = x_train.select_dtypes(exclude=np.number).columns.tolist()
#print(Table.loc[bad_indices, ['Area']])
#numeric_features = x_train.select_dtypes(include = 'number').columns.tolist()
#categorical_features = x_train.select_dtypes(exclude = 'number').columns.tolist()
#print("Numeric features:", numeric_features)
#print("Categorical features:", categorical_features)
for col in ['Parking','Elevator','Warehouse'] :
    x_train[col] = x_train[col].astype(int)
    x_test[col] = x_test[col].astype(int)
#print(x_train[['Parking', 'Elevator', 'Warehouse']].head())
#print(x_train.dtypes)
#Table = Table.drop(index=bad_indices, errors='ignore')
numeric_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2 , include_bias=False))
])
categoirical_pipeline = Pipeline([
     ('imputer' , SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
  ('numeric' , numeric_pipeline , numeric_features),
  ('categorial', categoirical_pipeline , categorical_features)
])
model = Pipeline([
     ('preprocessing', preprocessor),
     ('regression', LinearRegression())
])
model.fit(x_train,y_train)
predictions = model.predict(x_test)
model_usd =clone(model)
model_usd.fit(x_train,y2_train)
predictions_usd = model_usd.predict(x_test)
fig ,(ax1, ax2) = plt.subplots(1,2 , figsize= (12,5))
fig.suptitle('Actual vs Predicted Price')
ax1.scatter(y_test, predictions, alpha=0.6)
ax1.plot([y_test.min(), y_test.max()],[y_test.min(),y_test.max()],'r--')
ax1.set_xlabel('Actual price')
ax1.set_ylabel('Predicted price')
ax1.set_title('Price in Toman')
ax2.scatter( y2_test, predictions_usd , color='g', alpha=0.6)
ax2.plot([y2_test.min(),y2_test.max()],[y2_test.min(),y2_test.max()],'b--')
ax2.set_xlabel('Actual price')
ax2.set_ylabel('Predicted Price (USD)')
ax2.set_title('Price in USD')
#print(x_train.dtypes)
#print(Table[['Price', 'Price(USD)']].dtypes)
plt.tight_layout()
plt.show()
print("R2:", r2_score(y_test, predictions))
print("R2:", r2_score(y2_test, predictions_usd))

