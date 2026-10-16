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
x_train , x_test ,y_train , y_test,y2_train,y2_test = train_test_split(x,y,y2 , test_size=0.2 , random_state=42)
numeric_features = x_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = x_train.select_dtypes(exclude=np.number).columns.tolist()
for col in ['Parking','Elevator','Warehouse'] :
    x_train[col] = x_train[col].astype(int)
    x_test[col] = x_test[col].astype(int)

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Actual vs Predicted House Prices')


n = 70
houses = np.arange(n)


actual_stem = ax1.stem(
    houses - 0.15,
    y_test.to_numpy()[:n],
    linefmt='b-',
    markerfmt='bo',
    basefmt=' '
)

predicted_stem = ax1.stem(
    houses + 0.15,
    predictions[:n],
    linefmt='r-',
    markerfmt='rx',
    basefmt=' '
)

actual_stem.markerline.set_label('Actual')
predicted_stem.markerline.set_label('Predicted')

ax1.set_xlabel('Test House Number')
ax1.set_ylabel('Price (Toman)')
ax1.set_title('Price in Toman')
ax1.legend()

actual_stem_usd = ax2.stem(
    houses - 0.15,
    y2_test.to_numpy()[:n],
    linefmt='g-',
    markerfmt='go',
    basefmt=' '
)

predicted_stem_usd = ax2.stem(
    houses + 0.15,
    predictions_usd[:n],
    linefmt='m-',
    markerfmt='mx',
    basefmt=' '
)

actual_stem_usd.markerline.set_label('Actual')
predicted_stem_usd.markerline.set_label('Predicted')

ax2.set_xlabel('Test House Number')
ax2.set_ylabel('Price (USD)')
ax2.set_title('Price in USD')
ax2.legend()

plt.tight_layout()
plt.show()
print("R2:", r2_score(y_test, predictions))
print("R2:", r2_score(y2_test, predictions_usd))

