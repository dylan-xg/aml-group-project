
from datetime import datetime
from pathlib import Path
import tkinter as tk
import time

import cv2 as cv
from deepface import DeepFace
import pandas as pd

from .. import video_processor as VP
from .state import FaceRecognitionState
from .settings import *


def run(VIDEO_SOURCE: int | str | Path) -> None:
	state = create_state()
	try:
		VP.process_video(
			capture_location=VIDEO_SOURCE,
			callback=lambda frame: video_callback(state, frame),
			display_config=VP.DisplayConfig.OpenCV(frametime=1 / 15)
		)

	finally:
		print("Exiting Application...")
		try:
			state.window.destroy()
		except tk.TclError:
			pass


def create_state() -> FaceRecognitionState:
	window = tk.Tk()
	window.title("Face Recognition Control")
	window.geometry("500x170")

	name_label = tk.Label(window, text="Name:")
	name_label.grid(row=0, column=0, padx=10, pady=10)

	name_entry = tk.Entry(window, width=30)
	name_entry.grid(row=0, column=1, padx=10, pady=10)

	status_text = tk.StringVar()
	status_text.set("Status: Ready")

	status_label = tk.Label(window, textvariable=status_text)
	status_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

	state = FaceRecognitionState(
		window=window,
		name_entry=name_entry,
		status_text=status_text
	)

	register_button = tk.Button(
		window,
		text="Register Unknown Face",
		command=lambda: register_unknown(state)
	)
	register_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

	return state


def video_callback(state: FaceRecognitionState, frame: cv.typing.MatLike) -> cv.typing.MatLike:
	state.current_frame = frame.copy()

	recognise_current_frame(state)

	try:
		state.window.update()
	except tk.TclError:
		raise KeyboardInterrupt

	return frame


def should_run_recognition(state: FaceRecognitionState) -> bool:
	if not state.recognition_enabled:
		return False

	if state.current_frame is None:
		return False

	current_time = time.time()

	if current_time - state.last_recognition_time < state.recognition_interval:
		return False

	state.last_recognition_time = current_time

	return True


def recognise_current_frame(state: FaceRecognitionState) -> None:
	if not should_run_recognition(state):
		return

	start_time = time.time()

	dfs = DeepFace.search(
		img=state.current_frame, # type: ignore
		model_name=RECOGNITION_MODEL,
		distance_metric=DISTANCE_METRIC,
		normalization=RECOGNITION_MODEL,
		database_type="postgres",
		search_method="exact",
		enforce_detection=False
	)

	process_search_result(state, dfs)

	elapsed_time = time.time() - start_time

	print(f"Recognition time: {elapsed_time:.3f} seconds")


def process_search_result(state: FaceRecognitionState, dfs: list[pd.DataFrame]) -> None:
	if len(dfs) == 0 or dfs[0].empty:
		handle_unknown(state)
		return

	best_match = dfs[0].iloc[0]

	best_match["distance"]
	best_match["threshold"]

	if "img_name" in best_match.index:
		matched_name = str(best_match["img_name"])
	elif "identity" in best_match.index:
		matched_name = str(best_match["identity"])
	else:
		matched_name = "Unknown match"

	confidence = None

	if "confidence" in best_match.index:
		confidence = float(best_match["confidence"])
	print("Distance:",best_match["distance"],"Threshold:",best_match["threshold"],"Confidence:",best_match["confidence"])

	liveness_result = check_liveness(state)

	emotion_result = detect_emotion(state)



	handle_match(state, matched_name, confidence, liveness_result, emotion_result)


def handle_match(state: FaceRecognitionState, matched_name: str, confidence: float | None = None, liveness_result: bool | None = None, emotion_result: str | None = None) -> None:
	state.last_match_name = matched_name
	state.last_match_confidence = confidence
	state.last_unknown = False
	state.registration_required = False

	if confidence is None:
		message = "Recognised: " + matched_name
	else:
		message = "Recognised: " + matched_name + " (" + str(confidence) + "%)"

	if liveness_result is True:
		message = message + " | Liveness: passed"
	elif liveness_result is False:
		message = message + " | Liveness: failed"
	else:
		message = message + " | Liveness: not checked"

	if emotion_result is not None:
		message = message + " | Emotion: " + emotion_result
	else:
		message = message + " | Emotion: not checked"

	set_status(state, message)

def handle_unknown(state: FaceRecognitionState) -> None:
	state.last_match_name = None
	state.last_match_confidence = None
	state.last_unknown = True
	state.registration_required = True

	set_status(state,"Unknown face - enter name and register")


def register_unknown(state: FaceRecognitionState) -> None:
	name = state.name_entry.get().strip()

	if name == "":
		set_status(state, "Enter a name first")
		return

	if state.current_frame is None:
		set_status(state, "No frame available")
		return

	image_path = save_current_frame(state, name)

	set_status(state, "Registering " + name + "...")

	result = DeepFace.register(
		img=str(image_path),
		img_name=name,
		model_name=RECOGNITION_MODEL,
		normalization=RECOGNITION_MODEL,
		database_type="postgres",
		enforce_detection=False
	)

	inserted = result.get("inserted", 1)

	state.registration_required = False
	state.last_unknown = False
	state.last_match_name = name

	set_status(state,"Registered " + str(inserted) + " face record for " + name)

	print("Registered image:", image_path)


##SAVES REGISTERED IMAGE ON COMPUTER AS IMAGE FOR TESTING
def save_current_frame(state: FaceRecognitionState, name: str) -> Path:
	person_folder = DB_PATH / name
	person_folder.mkdir(parents=True, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	image_path = person_folder / (timestamp + ".jpg")

	cv.imwrite(str(image_path), state.current_frame) # type: ignore

	return image_path


#HELPER FUNCTION, CALL THIS WHEN CHANGING THE STATUS IN THE UI WINDOW
def set_status(state: FaceRecognitionState, text: str) -> None:
	state.status_message = text
	state.status_text.set("Status: " + text)
	state.window.update_idletasks()


#DOES NOTHING FOR NOW, FOR LIVENESS DETECTION
def check_liveness(state: FaceRecognitionState) -> bool | None: #RETURNS TRUE OR FALSE
	if not state.liveness_enabled:
		state.liveness_passed = False
		return None

	state.liveness_passed = True
	return True


#DOES NOTHING FOR NOW, FOR EMOTION DETECTION
def detect_emotion(state: FaceRecognitionState) -> str | None:
	if not state.emotion_enabled: #IF NOT TRUE
		state.last_emotion = None #RETURN NOTHING
		return None

	state.last_emotion = None #IF SOMEHOW TRUE, RETURN NOTHING
	return None
