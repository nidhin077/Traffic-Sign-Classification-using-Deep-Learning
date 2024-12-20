# importing requirement models
import sys
import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response, Response
from werkzeug.exceptions import BadRequest
import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
import numpy as np
from flask import Flask, render_template, Response, request, redirect, flash, url_for
# create flask app
app = Flask(__name__)

# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {}

# create a list of keys to use them in the select part of the html code
listOfKeys = []

# write the interface function
def get_prediction(img_bytes,model):
    img = Image.open(io.BytesIO(img_bytes))

    # inference
    results = model(img,size= 640)
    return results

def detect_live(model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert frame to PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        # run inference
        results = model(pil_img,size= 640)
        results.render()

        #encode image and yield it
        for img in results.ims:
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ret, buffer = cv2.imencode('.jpg', RGB_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

classes = { 0:'Speed limit (5km/h)',
			1:'Speed limit (15km/h)',
			2:'Dont Go straight',
			3:'Dont Go Left',
			4:'Dont Go Left or Right',
			5:'Dont Go Right',
			6:'Dont overtake from Left',
			7:'No Uturn',
			8:'No Car',
			9:'No horn',
			10:'Speed limit (40km/h)',
			11:'Speed limit (50km/h)',
			12:'Speed limit (30km/h)',
			13:'Go straight or right',
			14:'Go straight',
			15:'Go Left',
			16:'Go Left or right',
			17:'Go Right',
			18:'keep Left',
			19:'keep Right',
			20:'Roundabout mandatory',
			21:'watch out for cars',
			22:'Horn',
			23:'Speed limit (40km/h)',
			24:'Bicycles crossing',
			25:'Uturn',
			26:'Road Divider',
			27:'Traffic signals',
			28:'Danger Ahead',
			29:'Zebra Crossing',
			30:'Bicycles crossing',
			31:'Children crossing',
			32:'Dangerous curve to the left',
			33:'Dangerous curve to the right',
			34:'Speed limit (50km/h)',
			35:'Unknown1',
			36:'Unknown2',
			37:'Unknown3',
			38:'Go right or straight',
			39:'Go left or straight',
			40:'Unknown4',
			41:'ZigZag Curve',
			42:'Train Crossing',
			43:'Under Construction',
            44:'Unknown5',
			45:'Speed limit (60km/h)',
			46:'Fences',
			47:'Heavy Vehicle Accidents',
			48:'Unknown6',
			49:'Give Way',
			50:'No stopping',
			51:'No entry',
			52:'Unknown7',
			53:'Unknown8',
			54:'Speed limit (70km/h)',
			55:'speed limit (80km/h)',
			56:'Dont Go straight or left',
			57:'Dont Go straight or Right',
			   
            }
models = load_model('Trafffic_sign.h5')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(128,128))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 128,128,3)

	predict_x=models.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return classes [classes_x[0]]    



@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   

    
@app.route("/performance")
def performance():
	return render_template('performance.html')   
    
@app.route("/chart")
def chart():
	return render_template('chart.html')   


    
@app.route("/preview", methods=['GET', 'POST'])
def preview():
	return render_template("preview.html")

    

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)

		predict_result = predict_label(img_path)
		print(predict_result)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route('/index',methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('detect_choice') == 'live':
            return Response(detect_live(dictOfModels[request.form.get('model_choice')]), mimetype='multipart/x-mixed-replace; boundary=frame')

        file = extract_img(request)
        img_bytes = file.read()

        #choice of the model
        results = get_prediction(img_bytes,dictOfModels[request.form.get('model_choice')])
        print(f"User selected model : {request.form.get('model_choice')}")

        #update results.image with boxes and labels
        results.render()

        #encoding the resulting image and render it
        for img in results.ims:
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_arr = cv2.imencode('.jpg',RGB_img)[1]
            response = make_response(im_arr.tobytes())
            response.headers['Content-Type'] = 'image/jpeg'
        return response
    else:
        # in the select we will have each key of the list in option
        return render_template('indexOG.html',len= len(listOfKeys), listOfKeys= listOfKeys)

def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest('missing file parameter!!!!')
    file = request.files['file']
    if file.filename == '':
        raise BadRequest('Given file is invalid...')
    return file

if __name__ == '__main__':
    print('starting yolov5 webservice....')
    #getting dir containing models from comand args (or default 'model_train')
    models_dir = 'models_train'
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    print(f'watcing for yolov5 models under {models_dir}...')
    for r, d, f in os.walk(models_dir):
        for file in f:
            if '.pt' in file:
                #example : file = 'model1.pt'
                #the path of each model: os.path.join(r,file)
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(r,file)
                print(f'Loading model{model_path} with path {model_path}...')
                dictOfModels[model_name] = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=False)
                model = dictOfModels[model_name]
                model.conf = 0.5


        for key in dictOfModels:
            listOfKeys.append(key)
    #starting app
    app.run(debug = True)
                
