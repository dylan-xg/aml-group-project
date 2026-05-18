
from dataclasses import dataclass
import tkinter as tk

import cv2 as cv


#IF VIDEO DOES NOT DISPLAY SET recognition_enabled TO FALSE
@dataclass
class FaceRecognitionState:
	window: tk.Tk
	name_entry: tk.Entry
	status_text: tk.StringVar

	current_frame: cv.typing.MatLike | None = None

	recognition_enabled: bool = True #TO SEE ANY ISSUES DEEPFACE MIGHT BE CAUSING
	last_match_name: str | None = None
	last_match_confidence: float | None = None
	last_unknown: bool = False

	registration_required: bool = False

	liveness_enabled: bool = False
	liveness_passed: bool = True

	emotion_enabled: bool = False
	last_emotion: str | None = None

	status_message: str = "Ready"

	last_recognition_time: float = 0.0
	recognition_interval: float = 1.5 #YOU MIGHT HAVE TO RAISE THIS IF CPU IS SLOW, VIDEO WILL NOT APPEAR
