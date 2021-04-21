import streamlit as st
from config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import seaborn as sns
import pickle

st.sidebar.header(PROJECT_NAME)
st.sidebar.write(AUTHOR)

testData = 'datasets/test.csv'
trainData = 'datasets/train.csv'

choice = st.sidebar.radio("Project Menu",MENU_OPTIONS)

def load_data_test(rows = None):
    data = pd.read_csv(testData,index_col='id')
    return data

def load_data_train(rows = None):
    data = pd.read_csv(trainData)
    return data

dataTest = load_data_test()
dataTrain = load_data_train()

if choice =='View data':

    st.title("View raw data")
    
    st.write('Train Dataset')
    st.write(dataTrain)

    st.write('Test Dataset')
    st.write(dataTest)

if choice =='View stats':
    st.title('View Statistics in Dataset')
    
    st.write('Train Dataset')
    describetrain = dataTrain.describe()
    st.write(describetrain)

    st.write('Test Dataset')
    describetest = dataTest.describe()
    st.write(describetest)

if choice =='Visualize':
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

    st.subheader("Pie chart comparision of mobile")
    dataTrain.n_cores.value_counts().head().plot(kind='pie')
    st.pyplot()

    st.subheader("Column Comparison in Dataset through jointplot")
    xcolm = st.selectbox("X axis : select a column from the dataset", dataTrain.columns)
    ycolm = st.selectbox("Y axis : select a column from the dataset", dataTrain.columns)
    sns.jointplot(x=xcolm, y=ycolm, data=dataTrain)
    st.pyplot()

if choice =='Prediction':
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

    #st.write(clock_speed, type(clock_speed))

    clicked = st.button("make mobile price Prediction")
    if clicked:
        with open('models/mobile_price_prediction.pk','rb') as f:
            model = pickle.load(f)

        if model:
            features = np.array([battery_power,blue,clock_speed,fc,dual_sim,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi])
            prediction = model.predict(features.reshape(1.-1))
            st.header("predicted Price")
            st.write('0(low cost), 1(medium cost), 2(high cost) and 3(very high cost)')
            st.success(prediction[0])

if choice =='About':
    st.title('About the project')
    #st.image('img.png')
    st.write("""Maximum accuracy achieved in this specific dataset is 92 %, and features selected are 
    battery_power, blue, clock_speed, fc, dual_sim, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, 
    px_height, px_width, ram, sc_h, sc_w, talk_time,  three_g, touch_screen,wifi.""")
    st.write("""Price in the Price is the most effective attribute of marketing and business. The very first question of costumer is about the price of items. All the costumers are first worried and thinks “If he would be able to purchase something with given specifications or not”. So to estimate price at home is the basic purpose of the work. This paper is only the first step toward the above-mentioned destination. Artificial Intelligence-which makes machine capable to answer the questions intelligently- now a days is very vast engineering field. Machine learning provides us best techniques for artificial intelligence like classification, regression, supervised learning and unsupervised learning and many more. Different tools are available for machine learning tasks like MATLAB, Python, Cygwin, WEKA etc. We can use any of classifiers like Decision tree, Naïve Bayes and many more. Different type of feature selection algorithms are available to select only best features and minimize dataset. This will reduce computational complexity of the problem. As this is optimization problem so many optimization techniques are also used to reduce dimensionality of the dataset. Mobile now a days is one of the most selling and purchasing device. Every day new mobiles with new version and more features are launched. Hundreds and thousands of mobile are sold and purchased on daily basis. So here the mobile price class prediction is a case study for the given type of problem i.e. finding optimal product. The same work can be done to estimate real price of all products like cars, bikes, generators, motors, food items, medicine etc.  
Many features are very important to be considered to estimate price of mobile. For example Processor of the mobile. Battery timing is also very important in todays busy schedule of human being. Size and thickness of the mobile are also important decision factors. Internal memory, Camera pixels, and video quality must be under consideration. Internet browsing is also one of the most important constraints in this technological era of 21st century. And so is the list of many features based upon those, mobile price is decided. So we will use many of above mentioned features to classify whether the mobile would be very economical, economical, expensive or very expensive.
""")

    



