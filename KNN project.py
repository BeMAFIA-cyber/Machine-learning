import pandas as pd
import sklearn 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
Plane = pd.read_csv('train.csv')
Plane = Plane.copy()
Plane= Plane.dropna(subset = ['Arrival Delay in Minutes'])
mappings = { 'Gender' :{'Male':0 , 'Female' : 1}, 
            'Type of Travel': {'Personal Travel' : 0 , 'Business travel':1},
            'Customer Type': {'Loyal Customer' : 0 , 'disloyal Customer':1},
            'satisfaction':{'neutral or dissatisfied':0 ,'satisfied':1}}
for column ,mapping in mappings.items():
 Plane[column] = Plane[column].map(mapping)
encoder = OneHotEncoder(sparse_output=False)
encoded_class= encoder.fit_transform(Plane[['Class']])
Plane[encoder.get_feature_names_out(['Class'])] = encoded_class
Plane= Plane.drop(columns = ['Class'])
Plane= Plane.drop(columns = ['Unnamed: 0'])
x = Plane.drop('satisfaction', axis=1)
y= Plane['satisfaction']
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size = 0.2 ,train_size= 0.8, random_state= 4 )
categorial_features= x_train.select_dtypes(exclude=np.number).columns.tolist()
numeric_features= x_train.select_dtypes(include=np.number).columns.tolist()
model = Pipeline([
        ('normalize' , StandardScaler()),
        ('knn',KNeighborsClassifier())
])
model.fit(x_train,y_train)
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
disp =ConfusionMatrixDisplay.from_predictions(y_test,predictions, display_labels=['Neutral/Dissatisfied', 'Satisfied'],cmap='Blues',values_format='d')
plt.title('KNN Confusion Matrix')
disp.ax_.set_title(f'KNN Confusion Matrix\nAccuracy: {accuracy:.2%}')
plt.tight_layout()
plt.show()
