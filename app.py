import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('model2.h5', 'rb'))

sex_d = {0: 'Female', 1: 'Male'}
pclass_d = {0: 'First', 1: 'Second', 2: 'Third'}
embarked_d = {0: 'Cherbourg', 1: 'Queenstown', 2: 'Southampton'}

def main():
    st.set_page_config(page_title='Would you survive the crash?')

    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image(image='https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/' +
             'filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG')
    
    st.subheader('Author')
    st.write('Tomasz Radzki s22132')

    with overview:
        st.title('Would you survive the crash?')

    with left:
        sex_radio = st.radio('Sex', list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio('Class', list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embarked_radio = st.radio('Port of embarkation', list(embarked_d.keys()), format_func=lambda x: embarked_d[x])
    
    with right:
        age_slider = st.slider('Age', value=50, min_value=1, max_value=100)
        sibsp_slider = st.slider('Number of siblings and/or spouses', min_value=0, max_value=8)
        parch_slider = st.slider('Number of parents and/or children', min_value=0, max_value=6)
        fare_slider = st.slider('Ticket fare', min_value=0, max_value=500, step=10)

    data = pd.DataFrame.from_dict({
        'Pclass': [pclass_radio],
        'Age': [age_slider],
        'SibSp': [sibsp_slider],
        'Parch': [parch_slider],
        'Fare': [fare_slider],
        'Embarked': [embarked_radio],
        'male': [sex_radio]
    })

    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.header('Would given person survive: {0}'.format('YES' if survival[0] == 1 else 'NO'))
        st.subheader('Prediction confidence: {0:.2f}%'.format(s_confidence[0][survival][0] * 100))

if __name__ == '__main__':
    main()
