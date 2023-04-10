import numpy as np
import pickle
import streamlit as st


filename = r'C:\Users\kanishk arya\Desktop\Breast_Cancer_Detection\breast_cancer_prediction_model.sav'
loaded_model = pickle.load(open(filename,'rb')) 

# creating a function for python

def breast_cancer_prediction(input_data):

    input_data = (19.81,22.15,130,1260,0.09831,0.1027,0.1479,0.09498,0.1582,0.05395,0.7582,1.017,5.865,112.4,0.006494,0.01893,0.03391,0.01521,0.01356,0.001997,27.32,30.88,186.8,2398,0.1512,0.315,0.5372,0.2388,0.2768,0.07615)

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are prediction for one data point

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return "The Breast cancer is Malignant"
    
    else:
        return "The Breast Cancer is Benign"
    

def main():

    #giving a title
    st.title("Diabetes Prediction Web App")

    # getting the input from the user
    # mean radius	mean texture	mean perimeter	mean area	mean smoothness	mean compactness	mean concavity	mean concave points	mean symmetry	mean fractal dimension	...	worst radius	worst texture	worst perimeter	worst area	worst smoothness	worst compactness	worst concavity	worst concave points	worst symmetry	worst fractal dimension
    radius = st.text_input("Radius")
    texture = st.text_input("Texture")
    perimeter = st.text_input("Perimeter")
    area = st.text_input("Area")
    smoothness = st.text_input("Smoothness")
    compactness = st.text_input("Compactness")
    concavity = st.text_input("Concavity")
    concave_points = st.text_input("Concave Points")
    symmetry = st.text_input("Symmetry")
    fractal_dimension = st.text_input("Fractal Dimension")
    worst_radius = st.text_input("Worst Radius")
    worst_texture = st.text_input("Worst Texture")
    worst_perimeter = st.text_input("Worst Perimeter")
    worst_area = st.text_input("Worst Area")
    worst_smoothness = st.text_input("Worst Smoothness")
    worst_compactness = st.text_input("Worst Compactness")
    worst_concavity = st.text_input("Worst Concavity")
    worst_concave_points = st.text_input("Worst Concave Points")
    worst_symmetry = st.text_input("Worst Symmetry")
    worst_fractal_dimension = st.text_input("Worst Fractal Dimension")

    # code for prediction
    diagnosis = ""

    # creating a button for prediction
    if st.button('BCP RESULT'):
        diagnosis = breast_cancer_prediction([radius,texture,perimeter,area,smoothness,compactness,concavity,concave_points,symmetry,fractal_dimension,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension])

    st.success(diagnosis)


if __name__ == '__main__':
    main()