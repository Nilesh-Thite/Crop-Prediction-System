import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv("Crop_recommendation.csv")

#print(df.head())

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state = 2000)

clf.fit(X_train,y_train)

prediction_test = clf.predict(X_test)

#print(prediction_test)

from sklearn import metrics
#print("accuracy=",metrics.accuracy_score(y_test,prediction_test))

#print(model.feature_importances_)
feature_list = list(X.columns)
feature_imp = pd.Series(clf.feature_importances_, index=feature_list).sort_values(ascending=False)
#print(feature_imp)

#crop1 = [40, 40, 20, 29.126, 100.255, 9.255, 1000.51]

#crop1 = np.array([crop1])

#print(model.predict(crop1))

pickle.dump(clf,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))