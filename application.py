import os
import sys
import time
import uuid
import datetime
import glob

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_dropzone import Dropzone
from PdfProcessing import pdf_roll, pdf2text, pdfvertical2text

print(sys.version)
app = Flask(__name__)

app.config.update(
    UPLOADED_PATH='./uploads',
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_CUSTOM='True',
    DROPZONE_ALLOWED_FILE_TYPE='image/*, .xml, .config, .pdf',
    DROPZONE_MAX_FILE_SIZE=500,
    DROPZONE_MAX_FILES=1,
    DROPZONE_TIMEOUT=600000, # 10 minutes
    DROPZONE_REDIRECT_VIEW='result'  # set redirect view
)

dropzone = Dropzone(app)

class ProcessSettings:
    def __init__(self):
        self.param_dict = {}

    def set(self, request):
        self.param_dict['process'] = request.form['process']
        if self.param_dict['process'] == "PDF回転":
            self.param_dict['p1'] = request.form['p1']
        elif self.param_dict['process'] == "PDFテキスト化":
            self.param_dict['p1'] = request.form['p1']
    
    def get_process_name(self):
        return self.param_dict['process'] 

    def upload(self, request):
        start_time = time.perf_counter()
        f = request.files.get('file')
        print("file name:"+f.filename)
        print(vars(f))
        filename = self.get_work_filename()
        f.save(filename)
        self.param_dict['file_name'] = filename
        current_time = time.perf_counter()
        print("upload = {:.3f}sec".format(current_time - start_time))
        print("filename:"+filename)

    def process(self):
        start_time = time.perf_counter()
        filename_result = self.get_result_filename()

        if self.param_dict['process'] == "PDF回転":
            filename_result += '.pdf'
            input_angle = int(self.param_dict['p1'])
            angle = 180
            if input_angle <= 135:
                angle = 90
            if input_angle > 225:
                angle = 270
            pdf_roll(self.param_dict['file_name'], angle, filename_result)

        elif self.param_dict['process'] == "PDFテキスト化":
            filename_result += '.txt'           
            if self.param_dict['p1'] == "横書き":                
                pdf2text(self.param_dict['file_name'], filename_result)
            if self.param_dict['p1'] == "縦書き":
                pdfvertical2text(self.param_dict['file_name'], filename_result)

        current_time = time.perf_counter()
        print(self.param_dict['process']+" processing time = {:.3f}sec".format(current_time - start_time))
        return filename_result
    
    def get_work_filename(self):
        fname = str(uuid.uuid4())
        return os.path.join(app.config['UPLOADED_PATH'], fname)

    def get_result_filename(self):
        dt_now = datetime.datetime.now()
        return os.path.join(app.config['UPLOADED_PATH'], dt_now.strftime('%Y%m%d_%H%M%S_%f'))

p_settings = ProcessSettings()

@app.route('/', methods=['POST', 'GET'])
def index():
    ac = request.endpoint
    print("endpoint:"+ac)
    if request.method == 'POST': 
        p_settings.set(request)
        print("process:"+p_settings.get_process_name())
        return redirect('/upload') 
        
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        p_settings.upload(request)
    return render_template('upload.html', pocess_name = p_settings.get_process_name())

@app.route('/result')
def result():
    result_file_name = p_settings.process()
    ext_without_period = os.path.splitext(result_file_name)[1][1:]
    print("extension:"+ext_without_period)
    if ext_without_period == 'png':
        return render_template('result.html', result_url = result_file_name)
    elif ext_without_period == 'pdf':
        return render_template('result_pdf.html', result_url = result_file_name)
    else:
        return render_template('result_text.html', result_url = result_file_name)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADED_PATH'], filename)

if __name__ == '__main__':
    for p in glob.glob(app.config['UPLOADED_PATH']+'/**', recursive=True):
        if os.path.isfile(p):
            #os.remove(p)
            #print('not removed : '+p)
            pass
    app.run(debug=True)    