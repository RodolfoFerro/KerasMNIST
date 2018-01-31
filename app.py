from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import model_util
import os


# Configure upload path and extensions:
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])
NAME = "Handwritten Digits Classifier"

# Create a Flask app:
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = model_util.load_model()


# Allowed file check-up:
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define routes:
@app.route('/')
def index():
    return render_template('content.html', name=NAME)


@app.route('/process')
def process():
    error = 0
    return render_template('process.html', name=NAME, error=error)


@app.route('/upload', methods=['POST'])
def upload():
    # Check for directories:
    target = "/".join([APP_ROOT, UPLOAD_FOLDER])
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    if request.method == 'POST':
        file = request.files['file_name']
        if file and allowed_file(file.filename):
            error = 0
            print(file)
            filename = secure_filename(file.filename)
            # print(filename.split("."))
            input_img = "input." + filename.split(".")[1]
            output_img = "output." + filename.split(".")[1]
            destination = "/".join([target, input_img])
            # print(destination)
            file.save(destination)

            # Processing and prediction:
            img, img_th, rects = model_util.process_img(input_img)
            out = model_util.predict(model, img, img_th, rects, output_img)

            # Render results:
            return render_template('results.html', name=NAME, error=error,
                                   input_img=input_img, output_img=output_img,
                                   prediction=out)
        else:
            error = 1
            return render_template('process.html', name=NAME, error=error)


# Main:
if __name__ == '__main__':
    app.run(debug=False)
