import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('https://github.com/Poornesh5656/Diabetes_prediction/blob/main/trained_model%20(1).sav','rb'))

def diabetes_prediction(input_data):
     
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    


def main():

    st.title('Diabetes prediction web app')

    Pregnancies=st.text_input('Number of pregnancies')
    Glucose=st.text_input('Glucose')
    BloodPressure=st.text_input('BloodPressure Value')
    SkinThickness=st.text_input('SkinThickness Value')
    Insulin=st.text_input('Isulin Level')
    BMI=st.text_input('BMI Value')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction Value')
    Age=st.text_input('Age of the person')

    diagnosis=''

    if st.button("Diabetes test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI, DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__=='__main__':
    main()
