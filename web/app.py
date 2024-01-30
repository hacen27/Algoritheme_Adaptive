from flask import Flask, flash, request, redirect, url_for, render_template
from statistics import mode
from warnings import filters


import matplotlib.pyplot as plt 

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





 
@app.route('/', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

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

    flash('Image est charger avec succes')
    preds,prc1,prc2 = makePrediction(file_names)
    Zipp=zip(file_names,preds,prc1,prc2)
    return render_template('index.html', RESULTAT=Zipp )

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)





@app.route('/fonctional', methods=['POST'])
def functional():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
   
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
         
        else:
            flash('Mettre un image de types  - png, jpg, jpeg, gif')
            return redirect(request.url)

    flash('Image est charger par succes')
   
    Zipp=zip(file_names,)
    return render_template('fonctional.html', RESULTAT=Zipp)




 


if __name__ == "__main__":
   app.run(debug=True)