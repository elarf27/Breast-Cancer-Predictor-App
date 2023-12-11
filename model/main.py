import pandas as pd
import pickle5 as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def get_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def predictor(data):

    x = data.drop(['diagnosis'], axis = 1)
    y = data['diagnosis']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(penalty='l2')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print('accuracy_socre:', accuracy_score(y_pred, y_test))
    print('classification_report: \n', classification_report(y_pred, y_test))


    return model, scaler


def main():
    data = get_data()
    model, scaler = predictor(data)

    with open('model/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('model/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

if __name__ == '__main__':
    main()







