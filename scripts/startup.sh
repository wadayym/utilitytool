#!/bin/sh
apt-get update
apt-get -y install tesseract-ocr
apt-get -y install tesseract-ocr-jpn
apt-get -y install poppler-utils
apt-get -y install poppler-data
#!apt-get -y install libopencv-dev
gunicorn --bind=0.0.0.0 --timeout 600 application:app