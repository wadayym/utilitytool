#!/bin/sh
apt-get update
pip install torch==2.2.2 torchvision==0.17.2 --no-cache-dir
#apt-get -y install tesseract-ocr-jpn
#apt-get -y install poppler-utils
#apt-get -y install poppler-data
gunicorn --bind=0.0.0.0 --timeout 600 application:app