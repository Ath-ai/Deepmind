import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Set up the Streamlit app
st.set_page_config(page_title="Text-to-Image Generator", page_icon="🖼️")
st.title("Text-to-Image Generator")

# Initialize the text-to-image pipeline
@st.cache_resource
def load_model():
    return pipeline("text-to-image", model="runwayml/stable-diffusion-v1-5")

generator = load_model()

# Create the text input for the prompt
prompt = st.text_input("Enter your image description:", "A beautiful sunset over the ocean")

# Create a button to generate the image
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Generate the image
        image = generator(prompt)
        
        # Convert the image to bytes
        img_byte_arr = io.BytesIO()
        image[0].save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Display the generated image
        st.image(img_byte_arr, caption="Generated Image", use_column_width=True)
        
        # Provide a download button for the image
        st.download_button(
            label="Download Image",
            data=img_byte_arr,
            file_name="generated_image.png",
            mime="image/png"
        )

# Add some information about the app
st.markdown("""
## How to use
1. Enter a description of the image you want to generate in the text box.
2. Click the "Generate Image" button.
3. Wait for the image to be generated (this may take a few seconds).
4. View the generated image and download it if desired.

Please note that the quality and accuracy of the generated image may vary depending on the input description.
""")
