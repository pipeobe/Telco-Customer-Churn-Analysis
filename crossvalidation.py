import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedkFold, kFold, train_test_split

data = pd.read_csv("Churn") # Create DataFrame variable with the dataset

data["Churn"] = data["Churn"].map({"No":0, "Yes":1})
data["gender"] = data["gender"].map({"Male":0, "Female":1})

# Create Yearly or monthly Contract column - Binary No(0) or yes(1)
data["Contract"] = data["Contract"].replace(['Month-to-month'], 0)
data["Contract"] = data["Contract"].replace(['One year', 'Two year'], 1)
# Create tech or no techsupport column - Binary No(0) or yes(1)
data["TechSupport"] = data["TechSupport"].replace(['No internet service', 'No'], 0)
data["TechSupport"] = data["TechSupport"].replace(['Yes'], 1)
# Create online backup or no backup column - Binary No(0) or yes(1)
data["OnlineBackup"] = data["OnlineBackup"].replace(['No internet service', 'No'], 0)
data["OnlineBackup"] = data["OnlineBackup"].replace(['Yes'], 1)


data["Contract"] = data["Contract"].astype(int)
data["TechSupport"] = data["TechSupport"].astype(int)
data["OnlineBackup"] = data["OnlineBackup"].astype(int)

# Drop columns that wont be involved in calc
data.drop(labels=["SeniorCitizen","Partner","Dependents","tenure","PhoneService", "MultipleLines","InternetService","OnlineSecurity","DeviceProtection","StreamingTV","StreamingMovies","PaperlessBilling","PaymentMethod"], axis = 1, inplace = True)

 # Create answer 
 y = data['Churn']
 #drop y for calc
 data = data.drop('Churn',axis = 1)
 
 x_train, x_test, y_train, y_test = train_test_split(data,y, test_size=0.2, random_state=42)
 
 models = []
 models.append(('LR', LogisticRegression()))

results = dict()
for name, model in models:
    kfold = kFold(n_splits=8, random_state=8)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results[name]=(cv_results.mean())
    
    print()
    print("name results.mean")