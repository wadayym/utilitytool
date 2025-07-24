#!/bin/sh
python -m pip install -r requirements.txt
python -m pip show Pillow
python -c "import PIL; print(PIL.__version__)"
gunicorn --bind=0.0.0.0 --timeout 600 application:app