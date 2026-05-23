
from pathlib import Path

# Make sure to start the postgresql daemon
# Add DEEPFACE_POSTGRES_URI='postgresql://postgres@localhost/deepface' to your .env file.
CUSTOM_MODEL = "MyModel" #YOUR MODEL'S NAME HERE, could be anything
RECOGNITION_MODEL = CUSTOM_MODEL
DISTANCE_METRIC = 'euclidean_l2'
DB_PATH = Path('data/faces_db')
