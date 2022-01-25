import config
import os
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
import keras
import numpy as np
import urllib.request as urlreq
import matplotlib.pyplot as plt
import shutil
from pylab import rcParams
from keras.models import load_model
from keras import backend as K

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = config.secret_key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def start_page():
    return render_template('landing.html')

@app.route('/result', methods=['GET', 'POST'])
def upload_file():
	output = False
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part.')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected.')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			#uploadFile = request.files['file']
			#image = cv2.imdecode(np.fromstring(uploadFile.read(), np.uint8), cv2.IMREAD_UNCHANGED)
			filename = secure_filename(file.filename)
			pathToFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(pathToFile)
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
			
			faceDetected = apply_model(image)

			os.remove(pathToFile)
			
			output = True
	
	return render_template('landing.html', output = output, faceDetected = faceDetected, init = True)

def apply_model(image):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

	haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
	haarcascade = "haarcascade_frontalface_alt2.xml"
	detector = cv2.CascadeClassifier(haarcascade)
	faces = detector.detectMultiScale(image_gray)

	if len(faces) == 0:
		return False

	LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
	LBFmodel = "lbfmodel.yaml"
	landmark_detector  = cv2.face.createFacemarkLBF()
	landmark_detector.loadModel(LBFmodel)
	_, landmarks = landmark_detector.fit(image_gray, faces)

	fig = plt.figure()

	for landmark in landmarks:
		for x,y in landmark[0]:
			cv2.circle(image_rgb, (int(x), int(y)), 1, (255, 255, 255), 20)

	plt.axis("off")
	plt.imshow(image_rgb)
	fig.savefig('saved.png')
	shutil.move("saved.png", "static/saved.png")
	return True
	
if __name__ == "__main__":
	app.run()


