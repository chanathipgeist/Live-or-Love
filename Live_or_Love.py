import streamlit as st
import pandas as pd
import pickle
import json

# Load model โหลด model ที่สร้างไว้
with open('adaboost_model.pkl', 'rb') as file:
    adaboost = pickle.load(file)

# Load the questions from the JSON file โหลด คำถาม
with open('questions.json', 'r') as file:
    questions = json.load(file)
    

# Main function to run the web app
def main():
    st.title('💔 Live or Love: Divorce Predictor Dataset 💍')
    st.write('อยากรู้หรือไม่ว่าคุณและคู่จะ "อยู่" หรือ "จาก"!!!')
    st.write('Please answer the following questions:')
    st.title('How to Play')
    st.write('ผู้เล่นจะต้องทำการตอบคำถามทั้ง 54 ข้อ สามารถเล่นได้ทุกเพศทุกวัย โดยคำถามเหล่านี้ ได้ถูกตั้งขึ้นจากการสัมภาษณ์คนมา 170 คน')
    st.write('เอาล่ะ! หากผู้เล่นพร้อมกันแล้ว ก็ไปตอบคำถามกันเลย!!!')
    st.title('Start!')
    # Create a dictionary to store answers ไว้เก็บคำตอบ
    answers = {}

    # Display the questions and collect answers
    for i, (atr_name, question) in enumerate(questions.items(), start=1):
        st.markdown(f'<span style="font-size: 24px; color: black; font-weight: bold;">Question {i}:</span>', unsafe_allow_html=True)
        st.write(question)
        answers[atr_name] = st.slider(f'Choose for Question:{i}', 0, 4, 2)

    # Add a button for submission
    if st.button('Submit'):
        st.write('Done!')

        # Ensure all attributes are included and default to 0 if not answered
        for i in range(1, 55):
            if f'Atr{i}' not in answers:
                answers[f'Atr{i}'] = 0  # Default value if not answered

        # Convert answers dictionary to DataFrame with one row
        user_input = pd.DataFrame([answers])
        st.write('User Input DataFrame:')
        st.write(user_input)

        # Make prediction using the AdaBoostClassifier model
        prediction = adaboost.predict(user_input)
        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.write('ยินดีด้วย! คุณมีโอกาศที่จะได้รักกับคู่ของคุณ รักษาเขาไว้ให้ดี ')
        else:
            st.write('เสียใจด้วย! คุณทั้งคู่มีโอกาสที่จะเลิกกัน แต่ไม่ต้องเสียใจไป เพราะนี่เป็นเพียงการคาดการณ์จาก AI เท่านั้น(Ac=99%)')

if __name__ == '__main__':
    main()
