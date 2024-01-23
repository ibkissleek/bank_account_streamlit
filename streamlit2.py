import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing


loaded_model = pickle.load(open('model.sav', 'rb'))

def prediction(data):

     df= pd.DataFrame(data)
     df.iloc[8].replace({'Head of Household': 1, 'Spouse': 2, 'Child':3, 'Parent':4, 'Other relative':5, 
                                                'Other non-relatives':6}, inplace=True)
     df.iloc[9].replace({'Married/Living together': 1, 'Single/Never Married': 2, 'Widowed':3, 
                                        'Divorced/Seperated':4, 'Dont know':5}, inplace=True)
     df.iloc[10].replace({'Primary education': 1, 'No formal education': 2, 'Secondary education':3, 'Tertiary education':4, 'Vocational/Specialised training':5,
                                    'Other/Dont know/RTA':6}, inplace=True)
     
     


     label = preprocessing.LabelEncoder()

     df.iloc[7] = label.fit_transform(df.iloc[7])
     df.iloc[4] = label.fit_transform(df.iloc[4])
     df.iloc[3] = label.fit_transform(df.iloc[3])
     df.iloc[0] = label.fit_transform(df.iloc[0])
     df.iloc[11] = label.fit_transform(df.iloc[11])
     num_data = df.drop([2, 1]).values.reshape(1, -1)

     pred = loaded_model.predict(num_data)


     if pred[0] == 0:
          return "Customer has or use a bank account"
     else:
        return "The customer does not have or use a bank account"
     

def main():

    st.title("Bank Account Predictive Model")
    country = st.text_input("Country interviewee is in")
    year = st.number_input("Year survey was done in")
    uniqueid = st.text_input("Unique identifier for each interviewee")
    location_type = st.text_input("Type of location: Rural, Urban")
    cellphone_access = st.text_input("If interviewee hasaccess to a cellphone: Yes, No")
    household_size = st.number_input("Number of people living in one house")          
    age_of_respondent = st.number_input("The age of the interviewee")
    gender_of_respondent = st.text_input("Gender of interviewee")
    relationship_with_head = st.text_input("The interviewee's relationship with the head of the household")
    marital_status = st.text_input("The marital status of the interviewee")
    education_level = st.text_input("Highest level of education")
    job_type = st.text_input("Type of job interview has")
    


    bank_account = " "


    if st.button("Result"):
        bank_account = prediction([country, year, uniqueid, location_type, cellphone_access, household_size, 
                                   age_of_respondent, gender_of_respondent, relationship_with_head, 
                                   marital_status, education_level, job_type])
        

    st.success(bank_account)


if __name__ == "__main__":
    main()

    
