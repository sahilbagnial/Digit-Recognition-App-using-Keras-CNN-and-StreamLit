import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
st.title('Digit Recognition')
# Loading Model
model=load_model('.\model')
# Writing Instructions
instruct='''
1) Draw Digits in the given space (saying Draw Image Here).\n
2) The rescaled image will be shown corresponding to it.\n
3) Tap "Recognize Digit!" button to get the predicted output of digit.\n
4) You can select the brush width from the side bar.\n
5) To predict the digit again, clear the existing digit and tap the button. 
'''
st.write(instruct)
# Creating Columns to compare side by side
c1,c2,c3=st.beta_columns(3)
# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Brush Stroke width: ", 1, 25, 6)
fill_color='#000000'
stroke_color='#FFFFFF'
background_color='#000000'
y_pred=None
result=None
# Create a canvas component
with c1:
    st.header("Draw Image Here")
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=background_color ,
        background_image=None,
        update_streamlit=True,
        width=150,
        height=150,
        drawing_mode='freedraw',
        key='canvas',
    )
    img_data=canvas_result.image_data
with c2:
    st.header('              '+'Rescaled Image')
    if  img_data is not None:
        img = cv2.resize(img_data.astype('uint8'),(28,28))
        rescaled = cv2.resize(img, (150, 150), interpolation=cv2.INTER_LINEAR)
        c2.image(rescaled)
with c3:
    st.markdown('<br><br><br><br>',unsafe_allow_html=True)
    if st.button('Recgonize Digit!'):
        X_test= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_test=np.divide(X_test,255)
        X_test=np.expand_dims(X_test,-1).astype('float32')
        X_test=np.expand_dims(X_test,0)
        #st.write(X_test.shape)
        y_pred=model.predict(X_test)
        st.success('Recognized')
        result=np.argmax(y_pred,axis=-1)
        output='Recgonized digit is '+str(result[0])+' with '+str(y_pred.max())+' probability.'
if y_pred is not None and result is not None:
    st.write(output)
    st.bar_chart(y_pred[0])
