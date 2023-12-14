import streamlit as st
from PIL import Image
from classification import classification_img
import cv2
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="VEDECT - Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


st.title("VEDECT - Vehicle Detection")
st.text("Object Detection with Roboflow YOLO v5")

col1, col2, col3 = st.columns(3)

tab1, tab2, tab3 = st.tabs(["Home", "About", "Detect here"])

with tab1:
   st.header("Detect Your Image Here and Find A Great Experience")
   st.image("https://images.unsplash.com/photo-1465447142348-e9952c393450?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8M3x8Y2FycyUyMHN0cmVldCUyMGRlc2t0b3AlMjBiYWNrZ3JvdW5kfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60", width=700)

with tab2:
   st.header("VEDECT")
   st.write("Nama aplikasi dari hasil deployment proyek ini adalah Vehicle Detection, karena website ini adalah hasil dari deteksi objek enam jenis kendaraan. Aplikasi ini dapat berjalan dengan baik dan berhasil menghasilkan suatu data yang berisi jumlah setiap jenis kendaraan yang lewat di sebuah jalan dengan tingkat akurasi deteksi objek rata-rata adalah 82.67%, dimana data tersebut dapat digunakan oleh pihak yang berwenang untuk mengatur strategi lalu lintas yang baik, sehingga diharapkan dari pembuatan proyek ini dapat memberikan dampak positif bagi seluruh pengguna jalan raya. Input pada website yang sudah dideployment ini adalah tampilan kendaraan yang melintasi jalan raya yang kemudian menghasilkan suatu prediksi dan output berupa nama label dan jumlah label disertai keterangan ramai atau tidaknya kendaraan yang melintasi jalan raya tersebut dengan maksimal jumlah lebih dari 10 kendaraan yang terdeteksi.")

with tab3:
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", 'png', 'jpeg'])
    print(uploaded_file)
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        #image = Image.open(uploaded_file)
        #st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.image(opencv_image, channels="BGR")
        st.write("")
        st.write("Classifying...")
        label = classification_img(opencv_image, 'best.pt')
        st.write(label)
        st.snow()