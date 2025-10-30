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
        # Load the scaler and model
        scaler = jb.load('scaler.pkl')
        model = jb.load('svc_model.pkl')
        
        # Debug: Show what we're loading
        st.write("✅ Scaler and model loaded successfully")
        
        # Create a complete feature array FIRST, then scale
        gender_mapping = {'Female': 0, 'Male': 1}
        gender_encoded = gender_mapping[gender]
        
        stageT_mapping = {'T1': 1, 'T2': 7, 'T3a': 2, 'T3b': 3, 'T3c': 4, 'T3d': 7, 'T4a': 5, 'T4b': 6}
        stageT_encoded = stageT_mapping[stageT]
        
        stageN_mapping = {'N0': 0, 'N1a': 10, 'N1b': 2, 'N1c': 9, 'N2a': 3, 'N2b': 4, 'N3a': 6, 'N3b': 7, 'N3c': 8}
        stageN_encoded = stageN_mapping[StagN]
        
        sphincter_mapping = {'Yes': 0, 'No': 1}
        sphincter_encoded = sphincter_mapping[sphincter]
        
        biopsy_mapping = {
            'Well differentiated adenocarcinoma': 8,
            'Moderately differentiated adenocarcinoma': 6,
            'poorly differentiated adenocarcinoma': 4,
            'Mucoid adenocarcinoma': 1
        }
        biopsy_encoded = biopsy_mapping[biopsy]
        
        TNT_mapping = {'Short course': 1, 'long course': 0}
        TNT_encoded = TNT_mapping[tnt_c]
        
        course_mapping = {'Regression': 0, 'Stationary': 1, 'Progression': 2}
        course_encoded = course_mapping[course]
        
        # ADD MISSING MAPPINGS
        antrp_mapping = {'Anterior': 0, 'posterior': 1, 'lateral': 2, 'All': 3}
        antrp_encoded = antrp_mapping[antorp]
        
        invasion_mapping = {'Yes': 1, 'No': 0}
        invasion_encoded = invasion_mapping[invasion]
        
        tnt_chemo_mapping = {'induction': 0, 'Consolidation': 1}
        tnt_chemo_encoded = tnt_chemo_mapping[tnt]
        
        # Create complete input array with ALL features
        input_data = np.array([[
            Age, length, distance, dimensions, quadrants_involved, bmi,
            gender_encoded, stageT_encoded, stageN_encoded, sphincter_encoded,
            biopsy_encoded, TNT_encoded, course_encoded, antrp_encoded,
            invasion_encoded, tnt_chemo_encoded
        ]])
        
        # Scale the complete input
        input_scaled = scaler.transform(input_data)
        
        # Debug information
        st.write(f"📊 Input shape: {input_scaled.shape}")
        st.write(f"🔍 Model type: {type(model)}")
        
        # Make prediction
        prediction_encoded = model.predict(input_scaled)[0]
        
        st.write(f"🎯 Raw prediction value: {prediction_encoded}")
        
        # Display result
        if prediction_encoded == 1:
            st.success('✅ Your patient mostly will get Complete pathological response')
        elif prediction_encoded == 0:
            st.warning('⚠️ Your patient mostly will get partial pathological response')
        else:
            st.error('❌ Unfortunately, Your patient mostly will not get pathological response')
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.write("Please check that all required files (scaler.pkl, svc_model.pkl) are in the correct directory.")
