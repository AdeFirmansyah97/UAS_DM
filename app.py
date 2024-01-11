import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
svm = pickle.load(open('GaussianNB.pkl','rb'))

#load dataset
data = pd.read_csv('Bank Customer Churn Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Bank')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Diabetes Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = [' ','GaussianNB']
option = st.sidebar.selectbox('Model',activities)
st.sidebar.header('Data Customer')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset PIMA Indian</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('churn',axis=1)
y = data['churn']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():

    age = st.sidebar.number_input('Enter your age: ')
    tenure = st.sidebar.slider('Tenure : ', 1, 10, 20)
    balance = st.sidebar.slider('Balance', 0, 1000, 1000000)
    products_number = st.sidebar.slider('Products Number', 0, 1000, 1000000)
    credit_card = st.sidebar.slider('Credit Card', 0, 1000, 1000000)
    active_member = st.sidebar.slider('Active Member', 0, 1000, 10000)
    estimated_salary = st.sidebar.slider('Estimated Salary', 0, 10000, 100000)
    country_germany = st.sidebar.slider('Country Germany', 0, 1000, 1000)
    country_spain = st.sidebar.slider('Country Spain', 0, 1000, 100000)
    gender_male = st.sidebar.slider('gender', 0, 10, 100000)
    credit_score = st.sidebar.slider("Credit SCore", 0, 10, 100000)

    
    user_report_data = {
        "credit_score": credit_score,
        "age" : age,
        "tenure" : tenure,
        "balance" : balance,
        "products_number" : products_number,
        "credit_card" : credit_card,
        "active_member" : active_member,
        "estimated_salary" : estimated_salary,
        "country_Germany" : country_germany,
        "country_Spain" : country_spain,
        "gender_Male" : gender_male,
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
naive_bayes_accuracy = accuracy_score(y_test, svm.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena heart'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(naive_bayes_accuracy*100)+'%')