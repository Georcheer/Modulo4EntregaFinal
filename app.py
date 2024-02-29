import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np

model = pickle.load(open('C:/Users/fresh/OneDrive/Escritorio/Modulo 4 PY/GitHub/LinearReg', 'rb'))

st.title('Modelo de Regresion Lineal')
st.write('## Modelo de prediccion de Salarios')

st.sidebar.header('Informacion Relevante')

st.write('### Anexamos el documento encas o ser quiera revisar la base de datos utilizada')
df = pd.read_csv('C:/Users/fresh/OneDrive/Escritorio/salary_prediction_data.csv')

st.dataframe(df)

st.write('### A continuación se agrega el codigo utilizado para la creacion del modelo')
st.code("""
        
X = df.drop('Salary', axis = 1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LinearReg = LinearRegression()
LinearReg.fit(X_train, y_train)
        
y_pred = LinearReg.predict(X_test)
        
        """)

#Funcion
def reporteprediccion():
    Education = st.sidebar.slider('Education',0,3)
    Experience = st.sidebar.slider('Experience',1,29)
    Location = st.sidebar.slider('Location',0,2)
    Job_Title = st.sidebar.slider('Job_Title',0,3)
    Age = st.sidebar.slider('Age',18,80)
    Gender = st.sidebar.slider('Gender',0,1)

    user_report = {
        'Education': Education,
        'Experience': Experience,
        'Location': Location,
        'Job_Title': Job_Title,
        'Age': Age,
        'Gender': Gender
    }
    report_data = pd.DataFrame(user_report, index = [0])
    return report_data

user_data = reporteprediccion()
st.header('')
st.write(user_data)

salary = model.predict(user_data)
st.subheader('Salario Estimado')
st.subheader(np.round(salary[0], 2))