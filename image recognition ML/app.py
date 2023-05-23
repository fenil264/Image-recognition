from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential

app = Flask(__name__, template_folder='templates')

model = load_model('model/mnistCNN.h5')

@app.route('/')##URL
def upload_file():
  return render_template('index.html')
  
@app.route('/uploader', methods = ['POST'])
def upload_image_file():
  if request.method == 'POST':
       img = Image.open(request.files['file'].stream).convert("L")
       #img = np.mean(img,axis=2)
       img = img.resize((28,28))
       im2arr = np.array(img)
       im2arr = im2arr.reshape(1,28,28,1)
       y_pred = model.predict_classes(im2arr)
       ans = str(y_pred[0])
       ##return 'Predicted Number: ' + str(y_pred[0])
       return render_template('result.html',number=ans)
    
if __name__ == '__main__':
    app.run(debug = True)