
import os
from pathlib import Path

CUSTOM_MODEL = "MyModel" #YOUR MODEL'S NAME HERE, could be anything

PRERECORDING_PATH = Path('data/recording.avi')

if 'SSH_CLIENT' in os.environ:
	print('Remote session detected. Using video file.')
	VIDEO_SOURCE = PRERECORDING_PATH
else:
	print('Local session detected. Using live webcam.')
	VIDEO_SOURCE = 0

# Make sure to start the postgresql daemon
os.environ['DEEPFACE_POSTGRES_URI'] = 'postgresql://postgres@localhost/deepface' ##CHANGED LINE TO BE MY DB, WITH PASSWORD!!!

RECORDING_FPS=30.0
OUTPUT_PATH = Path('data/output.avi')

RECOGNITION_MODEL = CUSTOM_MODEL
DISTANCE_METRIC = 'euclidean_l2'

DB_PATH = Path('data/faces_db')
IMAGE1_PATH = Path('data/faces_db/RealName/YOUR_IMAGE1.jpg') #CHANGED TO MY NAME FOR FILE & FILE PATH
IMAGE2_PATH = Path('data/faces_db/james/YOUR_IMAGE2.jpg') #CHANGED TO MY NAME FOR FILE & FILE PATH
IMAGE3_PATH = Path('data/faces_db/james/YOUR_IMAGE3.jpg') #CHANGED TO MY NAME FOR FILE & FILE PATH
IMAGE2_PATH = IMAGE1_PATH #remove these if you want
IMAGE3_PATH = IMAGE1_PATH #same as above
