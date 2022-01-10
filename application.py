import os
import sys
import time
import uuid
import datetime
import glob
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
from PdfProcessing import pdf_roll, pdf2text, pdfvertical2text

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

class NumberPlace:
    def __init__(self):
        self.number_table = np.zeros((9, 9), dtype=np.int)

    def set(self, i, j, value): 
        self.number_table[i][j] = value
    
    def get(self): 
        return self.number_table, self.input_table
    
    def check_all(self):
        self.input_table = np.copy(self.number_table) 
        print(self.number_table) 
        result = self.check(0)
        return result

    def check(self, n):
        if n >= 9 * 9:
            return True
        i = n // 9
        j = n % 9
        l = self.number_table[i][j]
        #print(i, j, l)

        if l != 0:
            self.number_table[i][j] = 0
            if self.check3(i, j, l):
                self.number_table[i][j] = l
                return self.check(n + 1)
            self.number_table[i][j] = l
            return False
        for k in range(1,10):
            #print(i, j, k)
            if self.check3(i, j, k):
                self.number_table[i][j] = k
                if self.check(n + 1):
                    return True
                else:
                    self.number_table[i][j] = 0
        return False

    def check3(self, i, j, k):
        if self.check_box(i, j, k):
            if self.check_row(i, j, k):
                if self.check_column(i, j, k):
                    return True
                else:
                    return False
        return False

    def check_box(self, i, j, k): 
        box_row = i // 3
        box_column = j // 3
        rs = box_row * 3
        cs = box_column * 3
        box_list = self.number_table[rs:rs + 3, cs:cs + 3]
        if k in box_list:
            return False
        else:
            return True
        
    def check_row(self, i, j, k): 
        row_list = self.number_table[i,:]
        if k in row_list:
            return False
        else:
            return True
        
    def check_column(self, i, j, k): 
        column_list = self.number_table[:,j]
        if k in column_list:
            return False
        else:
            return True

class ProcessSettings:
    def __init__(self):
        self.param_dict = {}
        self.lineCounter = 0

    def clearLineCounter(self):
        self.lineCounter = 0

    def addLineCounter(self):
        self.lineCounter += 1
        if self.lineCounter == 10000:
            self.lineCounter = 0
        return self.lineCounter

    def set(self, request):
        self.param_dict['process'] = request.form['process']
        if self.param_dict['process'] == "PDF回転":
            self.param_dict['p1'] = request.form['p1']
        elif self.param_dict['process'] == "PDFテキスト化":
            self.param_dict['p1'] = request.form['p1']
        elif self.param_dict['process'] == "NumberPlace":
            pass
    
    def get_process_name(self):
        return self.param_dict['process'] 

    def upload(self, request):
        start_time = time.perf_counter()
        f = request.files.get('file')
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
        
    def printDictPretty(self, d, indent=0, file=sys.stdout):
        indent_word = '    '
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, (dict, list)):
                    print("{:4d} ".format(self.addLineCounter()) + indent_word * indent + str(key), file=file)
                    self.printDictPretty(value, indent+1, file)
                else:
                    print("{:4d} ".format(self.addLineCounter()) + indent_word * indent + str(key) + ':' +  str(value), file=file)
        else:
            for value in d:
                if isinstance(value, (dict, list)):
                    self.printDictPretty(value, indent, file)
                else:
                    print("{:4d} ".format(self.addLineCounter()) + indent_word * indent + str(value), file=file)    

p_settings = ProcessSettings()

PlaceName = [['00'] * 9 for i in range(9)]
for i in range(9):
    for j in range(9):
        PlaceName[i][j] = str(i * 10 + j)
bComplete = False


@app.route('/', methods=['POST', 'GET'])
def index():
    ac = request.endpoint
    print("endpoint:"+ac)
    if request.method == 'POST': 
        p_settings.set(request)
        if p_settings.get_process_name() =='NumberPlace':
            return redirect('/numberplace')
        return redirect('/upload') 
    return render_template('index.html')

@app.route('/numberplace', methods=['POST', 'GET'])
def numberplace():
    bComplete = False
    return render_template('numberplace.html', PlaceName = PlaceName, bComplete = bComplete)

@app.route('/resolve', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        numberPlace = NumberPlace()
        for i in range(9):
            for j in range(9):
                numberPlace.set(i, j, int(request.form[PlaceName[i][j]]))
        numberPlace.check_all()
        outTable, inTable = numberPlace.get()
        bComplete = True
        return render_template('numberplace.html', PlaceName = PlaceName, IN_Table = inTable, NP_Table = outTable, bComplete = bComplete)
    else:
        return redirect(url_for('numberplace'))

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        p_settings.upload(request)
    return render_template('upload.html', pocess_name = p_settings.get_process_name())

@app.route('/result')
def result():
    result_file_name = p_settings.process()
    ext_without_period = os.path.splitext(result_file_name)[1][1:]
    print(ext_without_period)
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
            os.remove(p)
            print('removed : '+p)
    app.run(debug=True)    