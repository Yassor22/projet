import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('master_sheet for machine learning.xlsx')
st.sidebar.header('This is a predictive model for locally advanced Rectal cancer to TNT')
st.sidebar.image('https://tse4.mm.bing.net/th/id/OIP.j2TJw0vapJPgHsqrhGYUHgHaHa?pid=ImgDet&w=185&h=185&c=7&dpr=1.1&o=7&rm=3')
st.sidebar.write('This application has been built to predict the response of locally advanced Rectal cancer to the types of total neoadjuvant therapy')
st.sidebar.write('The used model is RandomForestClassifier model with accuracy_score 78')
st.sidebar.write('Limitation : small-sized data\n If you can share deidentified data please Contact me')
st.sidebar.write('Created by ')
st.sidebar.markdown(' "Yasser Ali Okasha"')

st.sidebar.write('Supervised by ')
st.sidebar.markdown('"Professor.khaled Madbouly"')
st.sidebar.write(' Assisted By " Dr Shahed"')
st.sidebar.write('Contact details ')
st.sidebar.write("Email: yasser.okasha@alexmed.edu.eg")

st.title('Prediction of locally advanced rectal cancer response to TNT')
a1, a2, a3 = st.columns(3)
a1.image('cancer.JPG')
a2.image('radiotherapy.JPG')
a2.image('Capture.JPG')
a3.image('surgery.JPG')

st.text('Please Fill the following parameters about your patient to predict the Response')
st.write('Demographic Data')
gender = st.selectbox('Gender', ['Male', 'Female'])
Age = st.slider('Age', 10, 108)
bmi = st.number_input('BMI')
st.write('DRE')
length = st.number_input('Anal Canal length in CM')
distance = st.number_input('Distance from anal verge in CM')
quadrants_involved = st.slider('Quadrants involved', 0, 4)
antorp = st.selectbox('Site', ['Anterior', 'posterior', 'lateral', 'All'])
invasion = st.selectbox('Invasion of surrounding structures', ['Yes', 'No'])

st.write('Pretreatment MRI findings')
stageT = st.selectbox('T stage', ['T1', 'T2', 'T3a', 'T3b', 'T3c', 'T3d', 'T4a', 'T4b'])
StagN = st.selectbox('N stage', ['N0', 'N1a', 'N1b', 'N1c', 'N2a', 'N2b', 'N3a', 'N3b', 'N3c'])
dimensions = st.number_input('tumour dimensions')
sphincter = st.selectbox('Sphincters involvement', ['Yes', 'No'])

st.write('Colonoscopic Biopsy')
biopsy = st.selectbox('Histopathological result of the Biopsy', 
                      ['Well differentiated adenocarcinoma', 'Moderately differentiated adenocarcinoma', 
                       'poorly differentiated adenocarcinoma', 'Mucoid adenocarcinoma'])

st.write('TNT details')
tnt_c = st.selectbox('TNT Radiation', ['Short course', 'long course'])
tnt = st.selectbox('TNT Chemotherapy', ['induction', 'Consolidation'])

st.write('Post-treatment MRI findings')
course = st.selectbox('Radiological Response', ['Regression', 'Stationary', 'Progression'])
stagepT = st.selectbox('Post T stage', ['T1', 'T2', 'T3a', 'T3b', 'T3c', 'T3d', 'T4a', 'T4b'])
StagpN = st.selectbox('Post N stage', ['N0', 'N1a', 'N1b', 'N1c', 'N2a', 'N2b', 'N3a', 'N3b', 'N3c'])
sphincterp = st.selectbox('Sphincters involvement', ['Involved', 'Spared'])

btn = st.button('Submit')

if btn:
    try:
        # Load the model only - no scaling
        model = jb.load('svc_model.pkl')
        
        st.write("✅ Model loaded successfully")
        
        # Create mappings
        gender_mapping = {'Female': 0, 'Male': 1}
        stageT_mapping = {'T1': 1, 'T2': 7, 'T3a': 2, 'T3b': 3, 'T3c': 4, 'T3d': 7, 'T4a': 5, 'T4b': 6}
        stageN_mapping = {'N0': 0, 'N1a': 10, 'N1b': 2, 'N1c': 9, 'N2a': 3, 'N2b': 4, 'N3a': 6, 'N3b': 7, 'N3c': 8}
        sphincter_mapping = {'Yes': 0, 'No': 1}
        biopsy_mapping = {
            'Well differentiated adenocarcinoma': 8,
            'Moderately differentiated adenocarcinoma': 6,
            'poorly differentiated adenocarcinoma': 4,
            'Mucoid adenocarcinoma': 1
        }
        TNT_mapping = {'Short course': 1, 'long course': 0}
        course_mapping = {'Regression': 0, 'Stationary': 1, 'Progression': 2}

        # Create input array WITHOUT scaling
        input_data = np.array([[
            Age, length, distance, dimensions, quadrants_involved,
            gender_mapping[gender], stageT_mapping[stageT], 
            stageN_mapping[StagN], sphincter_mapping[sphincter],
            biopsy_mapping[biopsy], TNT_mapping[tnt_c], course_mapping[course]
        ]])
        
        # DEBUG INFORMATION
        st.write("---")
        st.write("🔍 **DEBUG INFORMATION:**")
        st.write(f"Input data shape: {input_data.shape}")
        st.write(f"Model expects: {model.n_features_in_} features")
        st.write(f"Input values: {input_data[0]}")
        
        # Check prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)[0]
            st.write(f"📊 Prediction probabilities: {probabilities}")
            st.write(f"📈 Class probabilities: {dict(zip(model.classes_, probabilities))}")
        
        # Get raw prediction
        prediction_encoded = model.predict(input_data)[0]
        st.write(f"🎯 Raw prediction value: {prediction_encoded}")
        
        # Check if model always predicts the same
        st.write(f"🏷️ Model classes: {model.classes_}")
        
        # Test with extreme values to see if prediction changes
        st.write("---")
        st.write("🧪 **TESTING EXTREME VALUES:**")
        
        # Test case 1: Young patient with early stage
        test_data1 = np.array([[
            30, 5, 3, 2, 1,  # Young, short length, close distance, small dimensions, few quadrants
            1, 1, 0, 1, 8, 1, 0  # Male, T1, N0, No sphincter, Well differentiated, Short course, Regression
        ]])
        
        # Test case 2: Old patient with advanced stage
        test_data2 = np.array([[
            80, 10, 8, 10, 4,  # Old, long length, far distance, large dimensions, all quadrants
            0, 6, 8, 0, 1, 0, 2  # Female, T4b, N3c, Sphincter involved, Mucoid, Long course, Progression
        ]])
        
        test_pred1 = model.predict(test_data1)[0]
        test_pred2 = model.predict(test_data2)[0]
        
        st.write(f"Test case 1 (Early stage): {test_pred1}")
        st.write(f"Test case 2 (Advanced stage): {test_pred2}")
        
        st.write("---")
        st.write("🎯 **PREDICTION RESULT:**")
        
        # Display result
        if prediction_encoded == 1:
            st.success('✅ Your patient mostly will get Complete pathological response')
        elif prediction_encoded == 0:
            st.warning('⚠️ Your patient mostly will get partial pathological response')
        else:
            st.error('❌ Unfortunately, Your patient mostly will not get pathological response')
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        import traceback
        st.write(f"Detailed error: {traceback.format_exc()}")
