
from pathlib import Path
from typing import Callable

import cv2


# Make sure to start the postgresql daemon
# Add DEEPFACE_POSTGRES_URI='postgresql://postgres@localhost/deepface' to your .env file.
CUSTOM_MODEL = "MyModel" #YOUR MODEL'S NAME HERE, could be anything
RECOGNITION_MODEL = CUSTOM_MODEL
DISTANCE_METRIC = 'euclidean_l2'
DB_PATH = Path('data/faces_db')

type Frame = cv2.typing.MatLike
"""A frame is a single image from a webcam or video input."""

type FrameDisplayCallback = Callable[[Frame], None]
"""A function used for a custom frame display output."""

type ProcessFrameCallback = Callable[[Frame], Frame | None]
"""A function that is called on a frame input for processing of some kind. Can return a modified result."""
