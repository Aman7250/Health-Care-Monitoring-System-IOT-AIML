from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from pymongo import MongoClient
import os
import cv2
import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import uuid
from predictions import predict
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename 
from keras.models import load_model
from keras.preprocessing import image
import cv2
import pickle


connection_string = os.environ.get('MONGODB_CONNECTION_STRING')
app = Flask(__name__)

model = pickle.load(open('Kidney.pkl', 'rb'))

model = load_model('ECG.h5')



UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
 
# MongoDB connection URI
mongo_uri = 'mongodb://localhost:27017'
client = MongoClient(mongo_uri)
db = client['my_database'] 

# Load pre-trained MobileNetV2 model for image classification
model = MobileNetV2(weights='imagenet')
model_path = 'new_data/img/trained_model.keras'  # Make sure this path is correct
model = load_model(model_path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_ct_image(image_path, output_path, target_size=(224, 224)):
    try:
        # Read the CT image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image to the target size
        resized_img = cv2.resize(img, target_size)

        # Save the resized image
        cv2.imwrite(output_path, resized_img)

        print(f"Image resized and saved to {output_path}")

        return resized_img

    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

@app.route('/detect', methods=['POST'])
def detect():
    collection = db['Heart_Prediction']  
    if 'image_file' not in request.files:
        return redirect('/heart')  # Redirect to the homepage or display an error message

    file = request.files['image_file']
    if file.filename == '':
        return redirect('/heart')  # Redirect to the homepage or display an error message

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Resize and preprocess the image
        resized_img = resize_ct_image(filepath, "static/images/resized_image.jpg", target_size=(224, 224))

        if resized_img is not None:
            # Convert grayscale image to 3 channels
            img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

            # Normalize pixel values to the range [0, 1]
            img = img / 255.0

            # Add batch dimension
            img = np.expand_dims(img, axis=0)

            # Make the prediction
            prediction = model.predict(img)

            # Interpret and get risk_level
            if prediction[0][0] >= 0.50:
                risk_level = "Your Heart is Healthy" 
            else:
                risk_level = "You have a Diseased Heart!!" 

            document = {
                'filename': filename,
                'risk_level': risk_level
            }
            collection.insert_one(document)


            return render_template('result.html', risk_level=risk_level, image_filename=filename) 
        else:
            return redirect('/heart.html')  # Redirect to the homepage or display an error message
    else:
        return redirect('/heart.html')  # Redirect to the homepage or display an error message

def calculate_risk_level(calcium_score):
    if calcium_score == 0:
        return "Very low risk"
    elif 1 <= calcium_score <= 50:
        return "Low risk"
    elif 51 <= calcium_score <= 250:
        return "Moderate risk"
    elif 251<= calcium_score <= 600:
        return "High risk"
    elif 601<= calcium_score <= 999:
        return "Very high risk"
    else :
        return "Dangerous!!!Seek Help Immediatly"

# Shubh
def classify(x):
    return model.predict(x)

@app.route("/classify", methods=["POST"])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, "uploads")
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        filepath = os.path.join(uploads_dir, f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        img = cv2.imread(filepath)
        resized_img = cv2.resize(img, (224, 224))

        x = np.expand_dims(resized_img, axis=0)

        pred = classify(x)
        y_class = np.argmax(pred)
        print("Classification:", y_class)

        index = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction',
                 'Premature Ventricular Contractions', 'Right Bundle Branch Block', 'Ventricular Fibrillation']
        result = index[y_class]

        return render_template("classify.html", prediction=result)
    return "No file uploaded."
    
@app.route('/result', methods=['GET', 'POST'])
def result():
    collection = db['Heart_Prediction'] 
    if request.method == 'POST':
        # Get the form data
        age = int(request.form['Age'])
        calcium_score = int(request.form['Calcium_score'])
        epicardial_volume = int(request.form['Epicardial_Tissue_Volume'])
        pericardial_volume = int(request.form['Pericardial_Tissue_Volume'])
        cardiac_fats = int(request.form['Sum_of_Cardiac_Fats'])
        
        # Calculate risk level based on calcium score
        risk_level = calculate_risk_level(calcium_score)

        prediction_data = {
            'age': age,
            'calcium_score': calcium_score,
            'epicardial_volume': epicardial_volume,
            'pericardial_volume': pericardial_volume,
            'cardiac_fats': cardiac_fats,
            'risk_level': risk_level
        }
        collection.insert_one(prediction_data)
        
        # Render the result template with the calculated risk level
        return render_template('result.html', risk_level=risk_level)

    # If it's a GET request, just render the form template
    return render_template('heart.html')

# Function to detect eyes in an image
def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if eye_cascade.empty():
        raise FileNotFoundError("Haar cascade file for eye detection not found.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return eyes

# Function to process uploaded image
# Function to process uploaded image
def process_image(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    eyes1 = detect_eyes(img1)
    eyes2 = detect_eyes(img2)
    
    if len(eyes1) > 0 and len(eyes2) > 0:
        (x1, y1, w1, h1) = eyes1[0]
        (x2, y2, w2, h2) = eyes2[0]
        eye_region1 = img1[y1:y1+h1, x1:x1+w1]
        eye_region2 = img2[y2:y2+h2, x2:x2+w2]
        eye_region2 = cv2.resize(eye_region2, (w1, h1))
        img1[y1:y1+h1, x1:x1+w1] = eye_region2
        
        # Generate a unique filename using uuid
        output_filename = str(uuid.uuid4()) + '.jpg'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        cv2.imwrite(output_path, img1)
        
        # Save the filename in the database
        collection = db['Eye_Prediction']
        document = {
            'filename': output_filename,
            'processed_image_path': output_path
        }
        collection.insert_one(document)
        
        return output_filename  # Return only the filename, not the full path
    else:
        return None

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/text')
def text():
    return render_template('text.html')

@app.route('/registration')
def registration():
    return render_template('registration.html')

@app.route('/uploads')
def uploads():
    return render_template('uploads.html')

@app.route('/output')
def output():
    return render_template('output.html')

@app.route("/ecg")
def about():
    return render_template("ecg.html")

@app.route("/first")
def ECG():
    return render_template("classify.html")


@app.route('/register', methods=['POST'])
def register():
    collection = db['users'] 
    user_data = request.json

    # Insert user data into MongoDB
    result = collection.insert_one(user_data)

    if result.inserted_id:
        return jsonify({'message': 'User registered successfully'}), 200
    else:
        return jsonify({'message': 'Error registering user'}), 500

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    collection = db['users'] 
    username = request.form['name']
    password = request.form['password']

    # Check if user exists in the database
    user = collection.find_one({'name': username, 'password': password})

    if user:
        # Redirect to a different page upon successful login
        return redirect(url_for('integration'))
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/integration')
def integration():
    return render_template('integration.html')

@app.route('/icu')
def icu():
    return render_template('icu.html') 

@app.route('/icu1')
def icu1():
    return render_template('icu1.html') 

@app.route('/heart')
def heart():
    return render_template('heart.html') 

@app.route('/riskfactor')
def riskfactor():
    return render_template('riskfactor.html') 

@app.route('/bone')
def bone():
    return render_template('bone.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def make_prediction():
    collection = db['Bone_Prediction']
    
    if request.method == 'POST':
        file = request.files['file']
        # Generate a unique filename using uuid
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
        filename = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            # Save the file
            file.save(filename)
            model = request.form['model']
            prediction = predict(filename, model)
            
            # Save the result image in the database
            document = {
                'filename': unique_filename,
                'prediction': prediction,
                'image_path': filename
            }
            collection.insert_one(document)
            
            return render_template('bone.html', prediction=prediction, image_file=filename)
        except Exception as e:
            return render_template('bone.html', error="Error processing image. Please upload a valid image file.", image_file=None)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    collection = db['Eye_Prediction'] 
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return redirect(request.url)
        file1 = request.files['file1']
        file2 = request.files['file2']
        if file1.filename == '' or file2.filename == '':
            return redirect(request.url)
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = 'input.jpg'
            filename2 = 'reference.jpg'
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            output_filename = process_image(os.path.join(app.config['UPLOAD_FOLDER'], filename1),
                                            os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            if output_filename:
                return render_template('output.html', output_image=output_filename)
            else:
                return "Eyes not detected in one or both images."
    return render_template('upload.html')
# < kidney >

@app.route('/kidney',methods=['GET'])
def Home():
    collection = db['Kidney_Prediction'] 
    return render_template('kidney.html')

@app.route("/predict_kd", methods=['POST'])
def predict_kd():
    collection = db['Kidney_Prediction'] 
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction = model.predict(values)

        return render_template('result_kd.html', prediction=prediction)
    

model = load_model('new_data/img/trained_model.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/detect_pneumonia', methods=['POST'])
def detect_pneumonia():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            result = "No Pneumonia Detected"
        else:
            result = "Pneumonia Detected"
        return render_template('result_pne.html', prediction=result)
    else:
        return 'Invalid file format. Please upload an image with .png, .jpg, or .jpeg extension.'


if __name__ == '__main__':
    app.run(debug=True)
