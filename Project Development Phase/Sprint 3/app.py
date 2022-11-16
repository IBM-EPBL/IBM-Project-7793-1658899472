import flask
from flask import request, render_template
from flask_cors import CORS
import joblib
import os
import cv2
from skimage import feature 

app = flask.Flask(__name__, static_url_path='')
CORS(app)

def quantify_image(image):
    features = feature.hog (image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return features

@app.route('/index', methods=['GET', 'POST'])
def sendHomePage():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predictSpecies():
    if request.method == 'POST':
        f=request.files['file'] 
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath, "uploads", f.filename)
        f.save(filepath)
        print("[INFO] Loading model...")
        dataset = request.form['dataset']
        if dataset=='spiral':
            m="C:/Users/Digant Gandhi/OneDrive/Desktop/Sprint4/parkinson.pkl"
        else:
            m="C:/Users/Digant Gandhi/OneDrive/Desktop/Sprint4/parkinson_wave.pkl"
        model = joblib.load(m)
        image = cv2.imread(filepath)
        output = image.copy()
        output = cv2.resize(output, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold (image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features= quantify_image(image)
        res=model.predict([features])
        if(res[0]):
            value="Parkinson" 
        else:
            value="Healthy"
    return render_template('predict.html',predict=value)

if __name__ == '__main__':
    app.run(debug=True)