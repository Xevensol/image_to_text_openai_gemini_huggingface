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
import openai
from openai import OpenAI
client = OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

# tilte of app 
st.markdown(
    """
    <style>
    .title {
        font-family: Arial, sans-serif;
        text-align: center;
        font-size: 36px;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>Image Analyzer</h1>", unsafe_allow_html=True)

st.markdown("<h6 style='font-family: Arial, sans-serif;'>This app for image Analyzing using the following given Models. Just select a model and see the result.</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='font-family: Arial, sans-serif;'> How it Works : </h6>", unsafe_allow_html=True)
st.markdown("<h6 style='font-family: Arial, sans-serif;'>1 . Select Model First <br> 2. Click on Browse select image <br> 3. Click on Analyze Button </h6>", unsafe_allow_html=True)


Model = st.radio("Select Model", ("Openai_vision_model", "Hugging_face_sales_force_model","Gemini-1.5-flash","claude-3-haiku"))

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
    
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

    if uploaded_image is not None:
        if uploaded_image.size > 20*1024*1024:
            st.error("Image size exceeds 20 MB. Please upload an image below 20 MB.")
            return
        image_data = uploaded_image.read()
        img = Image.open(BytesIO(image_data))
        st.image(img, caption='Uploaded Image', use_column_width=True)
# here we call openai vision model 
        if Model == "Openai_vision_model":
            if st.button('Analyze'):
                response = analyze_image(f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}")

                st.write("Description:", response)
# here we call hugging face saleforce model
        if Model == "Hugging_face_sales_force_model":
            if st.button("Analyze"):
                response = hugging_face_sales_force(image_data)
                st.write(response)
# here we call gemini pro vision model
        if Model == "Gemini-1.5-flash":
            if st.button("Analyze"):
                img = PIL.Image.open(uploaded_image)
                response = gemini_pro_flash(img)
                st.write(response)
        # if Model == "claude-3-haiku":
        #     if st.button("Analyze"):
        #         img = PIL.Image.open(uploaded_image)
        #         response = cloud_img(img)
        #         st.write(response)





# hugging face model
def hugging_face_sales_force(image_data):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer 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"}
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


## claude _ 3 
# def cloud_img(img):
#     message = client.messages.create(
#     model="claude-3-haiku-20240307",
#     max_tokens=1024,
#     messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",  # Specify that this is an image
#                         "image": img  # Include the image data
#                     },
#                     {
#                         "type": "text",
#                         "text": """Describe the given image."""
#                     }
#                 ],
#             }
#         ],
#     )

# # Return or process the message as needed
#     data = message.content[0].text
#     return data





if __name__ == "__main__":
    main()
