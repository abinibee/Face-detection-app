import cv2
import streamlit as st

st.markdown("<h1 style = 'color: #FFACAC'>FACE DETECTION APP</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; color: #F2921D'>Built by Gbenga Olaosebikan</h6>", unsafe_allow_html = True)

# Add an image to the page
st.image('img3.jpeg', caption = 'FACE DETECTOR', width = 200)

# Create a line and a space underneath
st.markdown('<hr><hr><br>', unsafe_allow_html= True)

# Add instructions to the Streamlit app interface to guide the user on how to use the app.
if st.button('Read the usage Instructions below'):
    st.success('Hello User, these are the guidelines for the app usage')
    st.write('Press the camera button for our model to detect your face')
    st.write('Use the MinNeighbour slider to adjust how many neighbors each candidate rectangle should have to retain it')
    st.write('Use the Scaler slider to specify how much the image size is reduced at each image scale')

st.markdown('<br>', unsafe_allow_html= True)
# Start the face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')
camera = cv2.VideoCapture(0)

# set the minNeighbours abd Scale Factor buttons
min_Neighbours = st.slider('Adjust Min Neighbour', 1, 10, 5)
Scale_Factor = st.slider('Adjust Scale Factor', 0.0, 3.0, 1.3)

st.markdown('<br>', unsafe_allow_html= True)

if st.button('FACE DETECT'):
# Initialize the webcam
    while True:
        _, camera_view = camera.read()   #....................................... Initiate the camera
        gray = cv2.cvtColor(camera_view, cv2.COLOR_BGR2GRAY) #.................. Grayscale it using the cv grayscale library
    #   Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor= Scale_Factor, minNeighbors= min_Neighbours, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    #   Draw rectangles around the detected faces
        for (x, y, width, height) in faces:
            cv2.rectangle(camera_view, (x, y), (x + width, y + height), (225, 255, 0), 2)
    # Display the camera_views
        cv2.imshow('Face Detection using Viola-Jones Algorithm', camera_view)
    # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    # Release the webcam and close all windows
    camera.release()
    cv2.destroyAllWindows()

# st.camera_input("Take a picture")
picture = st.camera_input('take a picture')
if picture:
    st.sidebar.image(picture, use_column_width= True, caption = f"welcome")
# let's assume the number of images gotten is 0
# img_counter = 0
# if k%256  == 32:
#     # the format for storing the images scrreenshotted
#     img_name = f'opencv_frame_{img_counter}'
#     # saves the image as a png file
#     cv2.imwrite(img_name, frame)
#     print('screenshot taken')
#     # the number of images automaticallly increases by 1
#     img_counter += 1

# --------------------------------------------------------------------------------------------------
# import streamlit as st
# import cv2
# import numpy as np

# # Function to detect faces in an image
# def detect_faces(image, scaleFactor, minNeighbors, rectangle_color):
#     # Load the Haar Cascade Classifier for face detection
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

#     # Draw rectangles around the detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color, 2)

#     return image

# # Streamlit app
# st.title('Face Detection App')

# # Add an image to the page
# st.image('img3.jpeg', caption = 'FACE DETECTOR', width = 200)


# # Add instructions
# st.write("Welcome to the Face Detection App!")
# st.write("1. Upload an image.")
# st.write("2. Adjust the parameters for face detection.")
# st.write("3. Choose the color for the detected face rectangles.")
# st.write("4. Click 'Detect Faces' to see the result.")
# st.write("5. Save the image with detected faces using the 'Save Image' button.")

# # Upload image
# uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_image:
#     # Display the uploaded image
#     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

#     # Get user inputs
#     scaleFactor = st.slider("Scale Factor", min_value=1.01, max_value=2.0, step=0.01, value=1.1)
#     minNeighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=5)
#     rectangle_color = st.color_picker("Rectangle Color", value="#FF5733")

#     if st.button("Detect Faces"):
#         # Read and process the image
#         image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
#         result_image = detect_faces(image, scaleFactor, minNeighbors, rectangle_color)

#         # Display the result image with detected faces
#         st.image(result_image, caption="Image with Detected Faces", use_column_width=True)

#         if st.button("Save Image"):
#             # Save the result image
#             cv2.imwrite("result_image.jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
#             st.write("Image with detected faces saved as 'result_image.jpg'.")

