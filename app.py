import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('/models/AIImageClassifier.hdf5')
    return model
model = load_model()

st.write("""

        # AI Image Classifier
         """
        )

file = st.file_uploader("Please upload an image", type = ['jpg', 'png'])
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
    size = (180,180)
    image = imageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    return prediction

if file is None:
    st.text('Please upload an image file')
    
else:
    image = Image.open(file)
    st.image = (image, use_column_width = True)
    predictions = import_and_predict(image, model)
    class_names = ['fake', 'real']
    string = "This image is most likely" + class_names[np.argmax(predictions)]
    st.success(string)
