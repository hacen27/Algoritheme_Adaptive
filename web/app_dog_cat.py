from flask import Flask, render_template, request
from statistics import mode
from warnings import filters
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
model1 = load_model("model_cats_dags.h5")

import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
 
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
    return render_template('dog_vs_cat.html')


# @app.route('/prediction', methods=["POST"])
def makePrediction(filenames):
    img = image.load_img(filenames,target_size = (150,150))
    x = image.img_to_array(img)
    x = x[np.newaxis]
    pred= model1.predict(x)
    return pred
   


   
@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    # if files == '':
    #     flash('No image selected for uploading')
    #     return redirect(request.url)
    files = request.files.getlist('files[]')
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
            #print('upload_image filename: ' + filename)
          
        else:
            flash('Mettre un image de types  - png, jpg, jpeg, gif')
            return redirect(request.url)

    flash('Image est charger par succes')
    preds = makePrediction(file_names)
    return render_template('dog_vs_cat.html', preds )
    # return render_template('index.html', filenames=file_names, datas=preds,pctgs1=prc1,pctgs2=prc2)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)






if __name__ == "__main__":
   app.run(debug=True)