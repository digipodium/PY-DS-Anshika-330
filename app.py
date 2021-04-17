import streamlit as st
from config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import seaborn as sns

st.sidebar.header(PROJECT_NAME)
st.sidebar.write(AUTHOR)

testData = 'test.csv'
trainData = 'train.csv'

choice = st.sidebar.radio("Project Menu",MENU_OPTIONS)

def load_data_test(rows = None):
    data = pd.read_csv(testData,index_col='id')
    return data

def load_data_train(rows = None):
    data = pd.read_csv(trainData)
    return data

dataTest = load_data_test()
dataTrain = load_data_train()

if choice =='view data':

    st.title("View raw data")
    
    st.write('Train Dataset')
    st.write(dataTrain)

    st.write('Test Dataset')
    st.write(dataTest)

if choice =='view stats':
    st.title('View Statistics in Dataset')
    
    st.write('Train Dataset')
    describetrain = dataTrain.describe()
    st.write(describetrain)

    st.write('Test Dataset')
    describetest = dataTest.describe()
    st.write(describetest)

if choice =='visualize':
    st.title("Graphs and charts")
    st.write('All visualization are on Training data')

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader("Histogram distribution in mobile devices")
    column = st.selectbox("select a column from the dataset", dataTrain.columns)
    bins = st.slider("select number of bins",5,100,20)
    histogram = dataTrain[column].plot.hist(bins=bins, title=f'{column} histogram analysis')
    st.pyplot()

    st.subheader('Scatter 3D plot on mobile devices')
    xcolumn = st.selectbox("select a first column from the dataset", dataTrain.columns)
    ycolumn = st.selectbox("select a second column for comparision", dataTrain.columns)
    zcolumn = st.selectbox("select a third column for comparosion", dataTrain.columns)
    fig = px.scatter_3d(dataTrain, x=xcolumn, y=ycolumn, z=zcolumn ,color='price_range',width=500,)
    st.plotly_chart(fig)

    st.subheader("Column Comparison through scatter plot")
    st.write(dataTrain.head())
    xcolumn = st.selectbox("select a column from the dataset for comparison", dataTrain.columns)
    ycolumn1 = st.selectbox("select a first column for comparision with first column selected", dataTrain.columns)
    ycolumn2 = st.selectbox("select a second column for comparosion with first column selected", dataTrain.columns)
    ycolumn3 = st.selectbox("select a third column for comparosion with first column selected", dataTrain.columns)
    plt.scatter(xcolumn, ycolumn1, data = dataTrain )
    plt.scatter(xcolumn, ycolumn2, data = dataTrain )
    plt.scatter(xcolumn, ycolumn3, data = dataTrain )
    plt.xlabel(xcolumn)
    plt.ylabel('mobile dataset columns')
    plt.title('Comparison of columns of Dataset 1')
    plt.grid(True)
    plt.legend()
    st.pyplot()

    st.subheader("Column Comparison in Dataset using 2 columns")
    st.header("Comparision Graph")
    xcol = st.selectbox("X axis :choose a column from the dataset", dataTrain.columns)
    ycol = st.selectbox("Y axis :choose a column from the dataset", dataTrain.columns)
    fig = px.scatter(dataTrain,x=xcol, y=ycol,color='price_range')
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Pie chart comparision of vehicle")
    dataTrain.n_cores.value_counts().head().plot(kind='pie')
    st.pyplot()

    st.subheader("Column Comparison in Dataset through jointplot")
    xcolm = st.selectbox("X axis : select a column from the dataset", dataTrain.columns)
    ycolm = st.selectbox("Y axis : select a column from the dataset", dataTrain.columns)
    sns.jointplot(x=xcolm, y=ycolm, data=dataTrain)
    st.pyplot()

if choice =='prediction':
    st.title('Use AI to predict')
    st.subheader('fill the detail and get result')
    
    battery_power =st.number_input('Total energy a battery can store in one time measured in mAh',min_value=501, max_value=1998)
    blue = st.number_input('Has bluetooth or not',min_value=0, max_value=1)
    clock_speed = st.number_input('speed at which microprocessor executes instructions',min_value=0.5000, max_value=3.0)
    fc = st.number_input('Front Camera mega pixels',min_value=0, max_value=19)
    dual_sim = st.number_input('Has dual sim support or not',min_value=0, max_value=1)
    four_g = st.number_input('Has 4G or not',min_value=0, max_value=1)
    int_memory = st.number_input('Internal Memory in Gigabytes',min_value=2, max_value=64)
    m_dep = st.number_input('Mobile Depth in cm',min_value=0.1000, max_value=1.0)
    mobile_wt = st.number_input('Weight of mobile phone',min_value=80, max_value=200)
    n_cores = st.number_input('Number of cores of processor',min_value=1, max_value=8)
    pc = st.number_input('Primary Camera mega pixels',min_value=0, max_value=20)
    px_height = st.number_input('Pixel Resolution Height',min_value=0, max_value=1960)
    px_width = st.number_input('Pixel Resolution Width',min_value=500, max_value=1998)
    ram = st.number_input('Random Access Memory in Mega Bytes',min_value=256, max_value=3998)
    sc_h = st.number_input('Screen Height of mobile in cm',min_value=5, max_value=19)
    sc_w = st.number_input('Screen Width of mobile in cm',min_value=0, max_value=18)
    talk_time = st.number_input('longest time that a single battery charge will last when you are',min_value=2, max_value=20)
    three_g = st.number_input('Has 3G or not',min_value=0, max_value=1)
    touch_screen = st.number_input('Has touch screen or not',min_value=0, max_value=1)
    wifi = st.number_input('as wifi or not',min_value=0, max_value=1)

    clicked = st.button("make mobile price Prediction")
    if clicked:
        st.success("abhi bahut code baki h")

if choice =='history':
    st.title('Previous prediction')

if choice =='about':
    st.title('About the project')
    #st.image('img.png')
    st.write("""Most user management systems have some sort of main page, usually known as a dashboard. You’ll create a dashboard in this section, but because it won’t be the only page in your application, you’ll also create a base template to keep the looks of the website consistent.
You won’t use any of Django’s advanced template features, but if you need a refresher on the template syntax, then you might want to check out Django’s template documentation""")



