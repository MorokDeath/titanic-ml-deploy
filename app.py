from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Загружаем и готовим данные
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return "Titanic ML model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df_new = pd.DataFrame([data])
    df_new['Sex'] = df_new['Sex'].map({'female': 1, 'male': 0})
    prediction = model.predict(df_new)[0]
    return jsonify({'survived': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
