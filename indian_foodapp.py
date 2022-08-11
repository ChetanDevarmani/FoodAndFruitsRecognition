from unicodedata import category
import streamlit as st
from PIL import Image
# from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import img_to_array,load_img
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

model = load_model('FV2.h5')
labels = {0: 'Apple',
 1: 'Banana',
 2: 'Bread',
 3: 'Onion',
 4: 'cabbage',
 5: 'capsicum',
 6: 'carrot',
 7: 'cauliflower',
 8: 'chapati',
 9: 'chicken 65',
 10: 'chicken curry',
 11: 'chilli pepper',
 12: 'chocolate cake',
 13: 'egg roll',
 14: 'eggs',
 15: 'french fries',
 16: 'ginger',
 17: 'grapes',
 18: 'ice cream',
 19: 'jalebi',
 20: 'kiwi',
 21: 'kulfi',
 22: 'lemon',
 23: 'mango',
 24: 'noodles',
 25: 'orange',
 26: 'pasta',
 27: 'pear',
 28: 'pineapple',
 29: 'pizza',
 30: 'potato',
 31: 'raddish',
 32: 'rice',
 33: 'sweetcorn',
 34: 'tomato',
 35: 'watermelon'}
food = ['Chapati', 'Chocolate cake', 'Pasta', 'Pizza','Kulfi','Jalebi', 'French fries','Chicken curry','Chicken 65', 'Eggs', 'Noodles','Ice cream','Rice','Egg roll','Bread']

fruits = ['Apple','Banana','Grapes','Kiwi','Lemon','Mango','Orange','Pear','Pineapple','Watermelon']
vegetables = ['Cabbage','Capsicum','Carrot','Cauliflower','Ginger','Onion','Potato','Raddish','Sweetcorn','Tomato','Chilli pepper']

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)



def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()

def run():
    st.title("🍔🍕Food Recognition for Dietary Assessment🥚🍗")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result= processed_img(save_image_path)
            print(result)
            if result in food:
                st.info('**Category : food**')
            elif result in fruits:
                st.info('**Category : fruits**')
            else:
                 st.info('**Category : vegetables**')
            st.success("**Predicted : "+result+'**')
            cal = fetch_calories(result)
            if cal:
                st.warning('**'+cal+'(100 grams)**')
run()