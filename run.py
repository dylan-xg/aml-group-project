
from pathlib import Path

import cv2 as cv

from src.panopticon import video_processor as VP
from src.panopticon.typing import Frame

# You will need to add your own video here
PRERECORDING_PATH = Path('data/testing/example.mp4')

def callback_function(frame: Frame) -> Frame:
	return cv.cvtColor(src=frame, code=cv.COLOR_RGB2GRAY)

# Forcing video file for testing currently.
VP.process_video(
	capture_location=PRERECORDING_PATH,
	callback=callback_function,
	display_config=VP.DisplayConfig.OpenCV(frametime=1/60)
)
