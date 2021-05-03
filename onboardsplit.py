import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import my data
df = pd.read_csv('signup_data.csv')

# Create new variables
df["today"] = pd.Timestamp("today").strftime("%Y-%m-%d")
df["dob"] = pd.to_datetime(df["dob"])
df["today"] = pd.to_datetime(df["today"])
df["signup"] = pd.to_datetime(df["signup"])

df["age"] = df["dob"].apply(lambda x : (pd.datetime.now().year - x.year))
df["customer_since"] = (df['today'] - df['signup']).dt.days
df = pd.get_dummies(df, columns = ['email', 'id_type', 'id_country', 'nationality', 'domicile', 'device'], drop_first = True)

# Set variables
X_var = np.asarray(df[['signupDurationSeconds', 'ad_redirect', 'vpn', 'creditScore', 'age', 'email_hotmail', 'email_yahoo', 'id_type_national_id', 'id_type_passport', 'id_country_Germany', 'id_country_Russia', 'id_country_UK',
 'nationality_Germany', 'nationality_Russia', 'nationality_UK', 'domicile_Germany', 'domicile_Russia', 'domicile_UK', 'device_ios', 'device_windows']])
y_var = np.asarray(df['top'])


#######################
# Random Forest Model
#######################
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, random_state=1, stratify=y_var)
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred_test = forest.predict(X_test)

# Accuracy
print(f"Accuracy Score is {accuracy_score(y_test, y_pred_test)}")
print(f"Confusion Matrix looks like: \n {confusion_matrix(y_test, y_pred_test)}")
print(f"Classification report: \n {classification_report(y_test, y_pred_test)}")
