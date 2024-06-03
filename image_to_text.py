import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from openai import OpenAI
import base64
import requests
import os 
import google.generativeai as genai
import PIL.Image

from openai import OpenAI
client = OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

method = st.radio("Select Method", ("Openai_vision_model", "Hugging_face_sales_force_model","gemini_pro_flash"))

# openai vision model 
def analyze_image(image_url):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url","image_url": {"url":image_url}}
                   
                ],
            }
        ],
        max_tokens=300,
    )
    description = response.choices[0].message.content
    return description

# Streamlit app image upload 
def main():
    st.title("Image Analyzer")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

    if uploaded_image is not None:
        if uploaded_image.size > 20*1024*1024:
            st.error("Image size exceeds 20 MB. Please upload an image below 20 MB.")
            return
        image_data = uploaded_image.read()
        img = Image.open(BytesIO(image_data))
        st.image(img, caption='Uploaded Image', use_column_width=True)
# here we call openai vision model 
        if method == "Openai_vision_model":
            if st.button('Analyze'):
                response = analyze_image(f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}")

                st.write("Description:", response)
# here we call hugging face saleforce model
        if method == "Hugging_face_sales_force_model":
            if st.button("Analyze"):
                response = hugging_face_sales_force(image_data)
                st.write(response)
# here we call gemini pro vision model
        if method == "gemini_pro_flash":
            if st.button("Analyze"):
                img = PIL.Image.open(uploaded_image)
                response = gemini_pro_flash(img)
                st.write(response)

# hugging face model
def hugging_face_sales_force(image_data):
    API_URL = os.getenv("API_URL")
    api = os.getenv("api_hug")
    headers = {"Authorization": api}
    data = image_data
    response = requests.post(API_URL, headers=headers, data=data)
    result = response.json()
    for res in result:
        for val in res.values():
            return val
# gemini pro vision 
def gemini_pro_flash(img):
    model = genai.GenerativeModel("gemini-1.5-flash")
    respon = model.generate_content(img)
    return respon.text

if __name__ == "__main__":
    main()
