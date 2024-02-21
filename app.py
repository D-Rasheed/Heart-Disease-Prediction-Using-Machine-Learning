import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# Importing data
df=pd.read_csv('heart.csv')


st.sidebar.image("https://i0.wp.com/thetechtian.com/wp-content/uploads/2022/07/Different-Sources-of-Energy.jpg?fit=1600%2C1067&ssl=1",width=300)
st.sidebar.header('HEART DISEASE PREDICTION')
menu = st.sidebar.radio(
    "Menu:",
    ("Intro", "Data", "Analysis", "Models"),
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Project Submitted By: Danish Rasheed')
st.sidebar.write('Matricola No.: VR497604')
st.sidebar.write('Github Repositories:')
st.sidebar.write('https://github.com/D-Rasheed/Heart-Disease-Prediction-Using-Machine-Learning')



if menu == 'Intro':
   st.image("https://www.nsmedicaldevices.com/wp-content/uploads/sites/2/2021/05/shutterstock_1303927084.png",width=700)
   st.title('HEART DISEASE PREDICTION DATASET')
   st.header('Context')
   st.write('Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.')
   st.header('Content')
   st.write('This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes.')
   st.write('''## Sources and References

kaggle: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Streamlit: https://streamlit.io/

Streamlit Doc: https://docs.streamlit.io/


## Used tools
| Data mining		| Visualization 	|
|---				|---				|
| - Jupyter Notebook| - Streamlit		|
| - Sklearn 		| - Python			|
| - Python			| - Numpy			|
| - Pandas			| - Matplotlib		|
| - Numpy			| - Seaborn		    |

''')

elif menu == 'Data':
   st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTidu96E25gtVXJ7bhUWgNE8_6jnIbeReG5UsoDpRBAjrBxlpnUPLl164mbO6M5VzVoLRs&usqp=CAU",width=700)
   st.title("DataFrame:")
   st.write(">***918 entries | 11 columns***")
   st.dataframe(df)
   # Calculate the count of missing values
   missing_values_count = df.isna().sum()
   # Display the result
   st.write("Count of Missing Values:")
   st.write(missing_values_count)
   
   min_age=df['Age'].min()
   max_age=df['Age'].max()
   Average_age=df['Age'].mean()
   st.write(f'The minimum age in our dataset : {min_age} years')
   st.write(f'The maximum age in our dataset : {max_age} years')
   st.write(f'The average age in our dataset : {Average_age} years')

   

elif menu == 'Analysis':
   st.write("update")
   label_dict = {1: 'Heart Disease', 0: 'Normal'}
   df['HeartDisease'] = df['HeartDisease'].map(label_dict)
   
   # Age distribution plot
   st.subheader('Age distribution')
   fig, ax = plt.subplots()
   sns.histplot(df, x='Age', ax=ax)
   plt.title('Age distribution')
   st.pyplot(fig)

   # Age and Heart Disease distribution
   st.subheader('Age and Heart Disease')
   fig, ax = plt.subplots()
   sns.histplot(data=df, x='Age', hue='HeartDisease', bins=20, ax=ax)
   plt.title('Age and Heart Disease')
   st.pyplot(fig)
 
   # Gender-wise average age
   st.subheader('Average age by Gender')
   st.write(df.groupby('Sex')['Age'].mean())

   # Distribution of Sex Feature
   st.subheader('Distribution of Sex Feature')
   st.write(df['Sex'].value_counts())
   fig, ax = plt.subplots()
   sns.countplot(x="Sex", data=df, ax=ax)
   plt.title('Distribution of Sex Feature')
   st.pyplot(fig)

   # Comparison of Heart disease between Male and Female
   st.subheader('Comparison of Heart disease between Male and Female')
   fig, ax = plt.subplots()
   sns.countplot(data=df, x='Sex', hue='HeartDisease', ax=ax)
   plt.title('Sex Vs Heart Disease')
   st.pyplot(fig)

   # Group data by Sex and ChestPainType and count occurrences
   st.subheader('Sex vs ChestPainType')
   df_grouped = df.groupby(['Sex', 'ChestPainType']).size().unstack(fill_value=0)
   fig, ax = plt.subplots(figsize=(10, 6))
   df_grouped.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
   plt.title('Sex vs ChestPainType')
   plt.xlabel('Sex')
   plt.ylabel('Count')
   plt.xticks(rotation=0)
   st.pyplot(fig)

   # Chest Pain and Heart Disease Relationship
   st.subheader('Chest Pain and Heart Disease Relationship')
   fig, ax = plt.subplots()
   sns.countplot(data=df, x="ChestPainType", hue='HeartDisease', ax=ax)
   plt.title('Chest Pain and Heart Disease Relationship')
   st.pyplot(fig)
   st.write("""
*)The plot indicates that patients experiencing Asymptomatic Chest Pain are more prone to developing Heart Failure.

*)Patients exhibiting typical Angina also face an high risk of Heart Failure.""")

   # Additional visualizations
   st.subheader('Cholesterol Distribution')
   fig, ax = plt.subplots()
   sns.histplot(data=df, x='Cholesterol', bins=20, hue='HeartDisease', ax=ax)
   plt.title('Cholesterol Distribution')
   st.pyplot(fig)

   st.subheader('RestingECG and Heart Disease Relationship')
   fig, ax = plt.subplots()
   sns.countplot(data=df, x='RestingECG', hue='HeartDisease', ax=ax)
   plt.title('RestingECG and Heart Disease Relationship')
   st.pyplot(fig)

   st.subheader('ExerciseAngina and Heart Disease Relationship')
   fig, ax = plt.subplots()
   sns.countplot(data=df, x='ExerciseAngina', hue='HeartDisease', ax=ax)
   plt.title('ExerciseAngina and Heart Disease Relationship')
   st.pyplot(fig)

   st.subheader('ST_Slope and Heart Disease Relationship')
   fig, ax = plt.subplots()
   sns.countplot(data=df, x='ST_Slope', hue='HeartDisease', ax=ax)
   plt.title('ST_Slope and Heart Disease Relationship')
   st.pyplot(fig)
   
   st.subheader('Oldpeak Distribution')
   fig, ax = plt.subplots()
   sns.histplot(data=df, x='Oldpeak', hue='HeartDisease', ax=ax)
   plt.title('Oldpeak Distribution')
   st.pyplot(fig)

   # Create a figure and axes for subplots
   st.subheader('Age vs RestingBP and RestingBP vs Cholesterol')
   fig, axs = plt.subplots(1, 2, figsize=(12, 6))

   # Plot Age vs RestingBP
   df.plot(kind='scatter', x='Age', y='RestingBP', s=32, alpha=.8, ax=axs[0])
   axs[0].set_title('Age vs RestingBP')

   # Plot RestingBP vs Cholesterol
   df.plot(kind='scatter', x='RestingBP', y='Cholesterol', s=32, alpha=.8, ax=axs[1])
   axs[1].set_title('RestingBP vs Cholesterol')

   # Show the plots
   st.pyplot(fig)

   # Create subplots for ExerciseAngina vs Age and RestingECG vs Age
   st.subheader('ExerciseAngina vs Age and RestingECG vs Age')
   fig, axes = plt.subplots(2, 1, figsize=(12, 12))

   # Plot ExerciseAngina vs Age
   sns.violinplot(data=df, x='Age', y='ExerciseAngina', inner='box', palette='Dark2', ax=axes[0])
   axes[0].set_title('ExerciseAngina vs Age')
   axes[0].set_ylabel('ExerciseAngina')
   axes[0].set_xlabel('Age')

   # Plot RestingECG vs Age
   sns.violinplot(data=df, x='Age', y='RestingECG', inner='box', palette='Dark2', ax=axes[1])
   axes[1].set_title('RestingECG vs Age')
   axes[1].set_ylabel('RestingECG')
   axes[1].set_xlabel('Age')

   # Show the plots
   st.pyplot(fig)

   # Create subplots for ChestPainType vs Age and Sex vs Age
   st.subheader('ChestPainType vs Age and Heart Disease vs Age')
   fig, axes = plt.subplots(2, 1, figsize=(12, 12))

   # Plot ChestPainType vs Age
   sns.violinplot(data=df, x='Age', y='ChestPainType', inner='box', palette='Dark2', ax=axes[0])
   axes[0].set_title('ChestPainType vs Age')
   axes[0].set_ylabel('ChestPainType')
   axes[0].set_xlabel('Age')

   # Plot Heart Disease vs Age
   sns.violinplot(data=df, x='Age', y='HeartDisease', inner='box', palette='Dark2', ax=axes[1])
   axes[1].set_title('Heart Disease vs Age')
   axes[1].set_ylabel('Heart Disease')
   axes[1].set_xlabel('Age')

   st.pyplot(fig)

elif menu =='Models':
   st.image("https://smartindustry.vn/wp-content/uploads/2020/02/what-is-deep-learning-large.jpg",width=700)
   # Streamlit app header
   st.title("MODEL SELECTION")
   #Adding a dropdown to select anyone model
   selected_model = st.selectbox("## Select Model", ["Logistic Regression", "SVM", "Random Forest","Xgboost"])
   
   
   ### LABEL ENCODING
   categorical_feat = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
   Numeric_feat = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

   from sklearn.preprocessing import LabelEncoder
   le =LabelEncoder()
   for i in categorical_feat:
    df[i]=le.fit_transform(df[i])

   #Separating the dependent feature and independent feature
   x=df.drop(columns='HeartDisease')
   y=df[['HeartDisease']]

   #STANDARDIZATION/FEATURE SCALING

   from sklearn.preprocessing import StandardScaler
   sc=StandardScaler()
   std_df=sc.fit_transform(x)
   
   #TRAIN TEST SPLITTING
   from sklearn.model_selection import train_test_split
   x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=4)

   #Import Evaluation Metrics
   from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, precision_recall_curve
  
   #Model training and evaluation
   #Logistic Regression
   if selected_model == "Logistic Regression":
        st.subheader('Logistic Regression')
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        lr_predict = lr.predict(x_test)
        st.write('Classification Report:')
        st.text(classification_report(y_test, lr_predict))

   #Confusion Matrix
        st.write('Confusion Matrix:')
        cm_lr = confusion_matrix(y_test, lr_predict)
        disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Heart Disease', 'Normal'])
        fig, ax = plt.subplots()
        disp_lr.plot(ax=ax)
        st.pyplot(fig)

   #Precision-Recall Curve
        st.write('Precision Recall Curve:')
        precision, recall, _ = precision_recall_curve(y_test, lr.predict_proba(x_test)[:, 1])
        fig, ax = plt.subplots()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve of Logistic Regression')
        st.pyplot(fig)

   #Support Vector Machines (SVM)
   if selected_model == "SVM":
        st.subheader('Support Vector Machines (SVM)')
        SVM = svm.SVC()
        SVM.fit(x_train, y_train)
        SVM_predict = SVM.predict(x_test)
        st.write('Classification Report:')
        st.text(classification_report(y_test, SVM_predict))

   #Random Forest Classifier
   if selected_model == "Random Forest":
        st.subheader('Random Forest Classifier')
        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        rfc_predict = rfc.predict(x_test)
        st.write('Classification Report:')
        st.text(classification_report(y_test, rfc_predict))

   #Confusion Matrix
        st.write('Confusion Matrix:')
        cm_rfc = confusion_matrix(y_test, rfc_predict)
        disp_rfc = ConfusionMatrixDisplay(confusion_matrix=cm_rfc, display_labels=['Heart Disease', 'Normal'])
        fig, ax = plt.subplots()
        disp_rfc.plot(ax=ax)
        st.pyplot(fig)
   
   #Precision-Recall Curve
        st.write('Precision Recall Curve:')
        precision, recall, _ = precision_recall_curve(y_test, rfc.predict_proba(x_test)[:, 1])
        fig, ax = plt.subplots()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve of Random Forest Classifier')
        st.pyplot(fig)

   #Feature Importance
        st.write('Feature Importance of Random Forest Classifier:')
        feature_importance = rfc.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig, ax = plt.subplots()
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(x_test.columns)[sorted_idx])
        plt.title('Feature Importance of Random Forest Classifier')
        plt.xlabel('Feature Importance')
        st.pyplot(fig)

   #XGBoost Classifier
   if selected_model == "Xgboost":
        st.subheader('XGBoost Classifier')
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(x_train, y_train)
        xgb_predict = xgb_classifier.predict(x_test)
        st.write('Classification Report:')
        st.text(classification_report(y_test, xgb_predict))

   #Confusion Matrix
        st.write('Confusion Matrix:')
        cm_xgb = confusion_matrix(y_test, xgb_predict)
        disp_xgboost = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Heart Disease', 'Normal'])
        fig, ax = plt.subplots()
        disp_xgboost.plot(ax=ax)
        st.pyplot(fig)