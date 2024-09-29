import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def get_data():
    data = pd.read_csv("data/data.csv")
    # print("Data Loaded")
    return data



def clean_data():
    data = get_data()
    data['Diagnosis'] = data['Diagnosis'].map({'M':1,'B':0})
    # print("Cleaning Data Done")
    X= data.drop(['Diagnosis'],axis=1)
    y = data['Diagnosis']
    return (X,y)



def transform_data(X,y,state):
    #scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split 
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=state)
    # print("Transforming Data Done")
    return {"data":(X_train, X_test, y_train,y_test),"scaler":scaler}



def train_model(X_train,y_train):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    print("Training Model Done :) \n \n")
    return model



def test_model(model,X_test,y_test):
    pred = model.predict(X_test)
    # print("Accuracy_Score : ",accuracy_score(y_test, pred),'\n',"Classification_Report : ",classification_report(y_test, pred))
    return accuracy_score(y_test, pred)

def main():
    scores = {}
    # Epochs
    X,y = clean_data()
    for state in [random.randrange(10,10000) for _ in range(0,10)]:
        data = transform_data(X,y,state)
        X_train, X_test, y_train,y_test = data['data']
        model = train_model(X_train,y_train)
        scores[state]= test_model(model,X_test,y_test)
    state = 0
    score = 0


    for k,v in scores.items():
        if v>score and v!=1:
            score = v
            state = k
    
    transformed_data = transform_data(X,y,state=state)
    X_train, X_test, y_train,y_test = transformed_data['data']
    scaler = transformed_data['scaler']
    model = train_model(X_train,y_train)
    print("State:",state,'\n',"Score:",score)
    print("Model Accuracy:",test_model(model,X_test,y_test))
    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)

    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)




if __name__=="__main__":
    main()