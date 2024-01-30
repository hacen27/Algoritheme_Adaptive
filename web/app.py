import base64
from flask import Flask, flash, request, redirect, url_for, render_template
from statistics import mode
from warnings import filters

from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt 

import cv2
import numpy as np
from sklearn.cluster import KMeans

import os

from PIL import Image



 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "cairocoders-ednalan"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')


def imageSegmentor(image_path,n_clusters=3,threshold=0):
    
    # Load image
    # image_path = "/content/drive/MyDrive/face_images/abdellahi.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Define a threshold
    # threshold = 100  # This threshold can be adjusted depending on the specific area and intensity desired

    # Apply threshold to get the pixels we are interested in
    mask = image > threshold

    # We use the mask to select the region of interest
    selected_pixels = image[mask].reshape(-1, 1)

    # Apply K-means clustering to the sthresholdelected pixels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(selected_pixels)

    # Map the pixel labels back to the original image shape
    segmented_image = np.zeros(image.shape, dtype=np.uint8)
    segmented_image[mask] = kmeans.labels_
    
    # Encode the segmented image as a JPEG
    _, encoded_image = cv2.imencode('.jpg', segmented_image)

    # Convert the encoded image to a base64 string
    base64_string = base64.b64encode(encoded_image).decode('utf-8')
    
    return base64_string



 
@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files[]')
    clustering = request.form.get('clustering')
    threshold = request.form.get('vol')

    file_names=[] 
    for file in files:   
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
            
            img = Image.open("static/uploads/"+ filename)
            
            if img.width > 190 or img.height > 150:
                output_size = (190, 150)
                img.thumbnail(output_size)
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          
        else:
            flash('Mettre un image de types  - png, jpg, jpeg, gif')
            return redirect(request.url)

    flash('Image est charger avec succes')
    
    
    Zipp=zip(file_names,clustering,vol)
    return render_template('index.html', RESULTAT=Zipp )

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
   app.run(debug=True)