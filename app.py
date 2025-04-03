from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
from PIL import Image
import numpy as np
import sqlite3
import sys
import os.path
import cv2
import base64
from io import BytesIO
import hashlib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask import current_app
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

# Load model
model = model_from_json(open("model_fer.json", "r").read())
# Load weights
model.load_weights('model_fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = None

detect_fn = tf.saved_model.load("Models/FaceDetector/saved_model")#Load the face detector
model = tf.keras.models.load_model("Models/FEC")#Load the facial emotion classifier

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}
static_files = ['display.css', 'eye.png', 'Picdetectb.jpg', 'thumbsup.jpg', 
                'github.png', 'IU.svg', 'UI.svg', 'RT.svg', 'UV.svg', 'VU.svg', 'feedback.svg']

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def bound(boxes, scores, h, w):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 1.5)

    # define a array as matrix
    signs = []
    for i in range(len(idxs)):
            signs.append(i)
    height, width = h, w
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            ymin = int((boxes[i][0] * height))
            xmin = int((boxes[i][1] * width))
            ymax = int((boxes[i][2] * height))
            xmax = int((boxes[i][3] * width))
            signs[i] = [ymin,ymax,xmin,xmax]
    return signs

def draw_bounding_box(frame, detect_fn):
    #Returns the coordinates of the bounding boxes.
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    h, w = frame.shape[:2]
    boxes = boxes.tolist()
    scores = scores.tolist()
    coordinates = bound(boxes, scores, h, w)
    return coordinates

def detectandupdate(img):
    path = "static/" + str(img)
    image = cv2.imread(path)
    coordinates = draw_bounding_box(image, detect_fn)

    #Loop over the each bounding box.
    for (y, h, x, w) in coordinates:
        cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
        img2 = image[y:h, x:w]#Get the face from the image with this trick.
        img2 = tf.image.resize(img2, size = [128, 128])#Input for the model should have size-(128,128)
        pred = model.predict(tf.expand_dims(img2, axis=0))
        pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
        #These conditions are just added to draw text clearly when the head is so close to the top of the image. 
        if x > 20 and y > 40:
            cv2.putText(image, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, pred_class, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    path2 = f"static/pred{img}"
    #Save as predimg_name in static.
    cv2.imwrite(path2, image)


    return ([img, "pred" + img])

def detect_emotion():
    global cap

    if cap is None:
        cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, jpeg = cv2.imencode('.jpg', test_img)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE,
              password_hash TEXT,
              email VARCHAR,
              phone_no INTEGER,
              R_address VARCHAR(255),
              gender VARCHAR,
              age INTEGER,
              dob DATE)''')

    conn.commit()
    conn.close()

init_db()

# Flag to stop the live detection
stop_detection = False


# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            
            # Hash the password for comparison
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, hashed_password))
            user = c.fetchone()
            conn.close()
            
            if user:
                # Successful login, redirect to home page
                return redirect('/home')
            else:
                # Invalid credentials, render login page with error message
                return render_template('login1.html', error='Invalid username or password')
        
        except Exception as e:
            # Handle any exceptions
            return render_template('error.html', message="An error occurred during login. Please try again later.")

    # If it's a GET request, render the login page
    return render_template('login1.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['U_password']
            email = request.form['email']
            phone = request.form['phone_no']
            R_address = request.form['R_address']
            gender = request.form['gender']
            age = request.form['age']
            dob = request.form['dob']

            # Hash the password
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password_hash, email, phone_no, R_address, gender, age, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, password_hash, email, phone, R_address, gender, age, dob))
            conn.commit()
            conn.close()

            return "Registration successful!", 200

        except Exception as e:
            print("Error during registration:", e)
            return "An error occurred during registration. Please try again later.", 500

    return render_template('register1.html')

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/live_detection')
def live_detection():
    return render_template('live_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
    return 'Emotion detection started.'

@app.route('/stop')
def stop():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return 'Emotion detection stopped.'

@app.route('/picdelete')
def picdelete():
    #When this function is called all the files that are not present in the
    #list static_files will be deleted.
    for file in os.listdir("static"):
        if file not in static_files:
            os.remove(f"static/{file}")
    return ("nothing")

@app.route('/detectpic', methods=['GET', 'POST'])
def detectpic():
    UPLOAD_FOLDER = 'static'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':

        file = request.files['file']

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result =detectandupdate(filename)
            return render_template('showdetect.html', orig=result[0], pred=result[1])

@app.route('/picdetect')
def picdetect():
    return render_template('picdetect.html')


if __name__ == '__main__':
    app.run(debug=True)
