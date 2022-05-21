import numpy as np
from tqdm import tqdm
import pandas as pd
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from flask import Flask


def auc_score(y_true, y_pred):
    if len(np.unique(y_true[:,1])) == 1:
        return 0.5
    else:
        return roc_auc_score(y_true, y_pred)
    
#AUC score
def auc( y_true, y_pred ) :
    score = tf.py_function( auc_score,[y_true, y_pred],'float32',name='sklearnAUC' )
    return score


best_model = keras.models.load_model('best_model_efficientnetb4_agumented', custom_objects={'auc':auc})

from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 102 * 102


dir_path = "images"

#best_model = keras.models.load_model('best_model8', custom_objects={'auc':auc})

import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    dic = {}    

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            #print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print(os.path.join("images",filename))
            image_string = tf.io.read_file("images"+ "\\" + filename)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, [204, 136])
            image = tf.expand_dims(image, axis=0)
            #print(best_model.predict(image))
            probabilities = best_model.predict(image)[0]
            labels = ['Healthy', 'Multiple Disease', 'Rust', 'Scab']
            a= [dic.update({i:j}) for i, j in zip(labels, probabilities)]
            ind = np.argmax(probabilities)
            if ind ==0:
                flash("The Tree is Healthy!")
            else:
                flash('The Tree is infected by '+labels[ind])

            return render_template('upload.html', filename=filename, dic = dic, score = None)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    if 'filepath' in request.form.to_dict():
        if request.form.to_dict()['filepath']=='':
            return redirect(request.url)
        data = pd.read_csv('train.csv')
        file_path_dict = request.form.to_dict()['filepath']
        path = file_path_dict
        images_in_cwd = os.listdir(path)
        images = [i.split(".")[0] for i in images_in_cwd]
        images = data[data['image_id'].isin(images)]['image_id'].values
        b = data[data['image_id'].isin(images)][["healthy", "multiple_diseases", "rust", "scab"]].values
        a = np.empty((0,4),float)
        for i in tqdm(images):
            image_string = tf.io.read_file(os.path.join(path, i)+'.jpg')
            image = tf.image.decode_jpeg(image_string, channels=3)
            # This will convert to float values in [0, 1]
            image = tf.image.convert_image_dtype(image, tf.float32)
            #resize the image
            image = tf.image.resize(image, [204, 136])
            #print(image.shape)
            image = tf.expand_dims(image, axis=0)
            a = np.concatenate((a,best_model.predict(image)), axis=0)
        score = roc_auc_score(b, a)    
        return render_template('upload.html', filename=None, dic = None, score = score)     
        
    

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()