
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2 as cv
import pandas as pd
import tkinter as tk
from deepface import DeepFace
from video_feed import VideoProcessor


DB_PATH = Path('data/faces_db')
RECOGNITION_MODEL = 'Facenet'
DISTANCE_METRIC = 'euclidean_l2'

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

	recognition_busy: bool = False
	last_recognition_time: float = 0.0
	recognition_interval: float = 1.5 #YOU MIGHT HAVE TO RAISE THIS IF CPU IS SLOW, VIDEO WILL NOT APPEAR

class FaceRecognitionController:

	@staticmethod
	def run(VIDEO_SOURCE: int | str | Path) -> None:
		state = FaceRecognitionController.create_state()

		try:
			VideoProcessor.process_video(
				capture_location=VIDEO_SOURCE,
				callback=lambda frame: FaceRecognitionController.video_callback(state, frame),
				frametime=1 / 15
			)

		finally:
			try:
				state.window.destroy()
			except tk.TclError:
				pass

	@staticmethod
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
			command=lambda: FaceRecognitionController.register_unknown(state)
		)
		register_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

		return state

	@staticmethod
	def video_callback(state: FaceRecognitionState, frame: cv.typing.MatLike) -> cv.typing.MatLike:
		state.current_frame = frame.copy()

		FaceRecognitionController.recognise_current_frame(state)

		try:
			state.window.update()
		except tk.TclError:
			raise KeyboardInterrupt

		return frame

	@staticmethod
	def should_run_recognition(state: FaceRecognitionState) -> bool:
		if not state.recognition_enabled:
			return False

		if state.current_frame is None:
			return False

		if state.recognition_busy:
			return False

		current_time = time.time()

		if current_time - state.last_recognition_time < state.recognition_interval:
			return False

		state.last_recognition_time = current_time

		return True

	@staticmethod
	def recognise_current_frame(state: FaceRecognitionState) -> None:
		if not FaceRecognitionController.should_run_recognition(state):
			return

		state.recognition_busy = True

		try:
			search_path = DB_PATH / "_current_search.jpg"
			cv.imwrite(str(search_path), state.current_frame) # type: ignore

			dfs = DeepFace.search(
				img=str(search_path),
				model_name=RECOGNITION_MODEL,
				distance_metric=DISTANCE_METRIC,
				normalization=RECOGNITION_MODEL,
				database_type="postgres",
				search_method="exact",
				enforce_detection=False
			)

			FaceRecognitionController.process_search_result(state, dfs)

		finally:
			state.recognition_busy = False

	@staticmethod
	def process_search_result(state: FaceRecognitionState, dfs: list[pd.DataFrame]) -> None:
		if len(dfs) == 0 or dfs[0].empty:
			FaceRecognitionController.handle_unknown(state)
			return

		best_match = dfs[0].iloc[0]

		if "img_name" in best_match.index:
			matched_name = str(best_match["img_name"])
		elif "identity" in best_match.index:
			matched_name = str(best_match["identity"])
		else:
			matched_name = "Unknown match"

		confidence = None

		if "confidence" in best_match.index:
			confidence = float(best_match["confidence"])

		liveness_result = FaceRecognitionController.check_liveness(state)

		emotion_result = FaceRecognitionController.detect_emotion(state)



		FaceRecognitionController.handle_match(state, matched_name, confidence, liveness_result, emotion_result)

	@staticmethod
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

		FaceRecognitionController.set_status(state, message)

	@staticmethod
	def handle_unknown(state: FaceRecognitionState) -> None:
		state.last_match_name = None
		state.last_match_confidence = None
		state.last_unknown = True
		state.registration_required = True

		FaceRecognitionController.set_status(state,"Unknown face - enter name and register")

	@staticmethod
	def register_unknown(state: FaceRecognitionState) -> None:
		name = state.name_entry.get().strip()

		if name == "":
			FaceRecognitionController.set_status(state, "Enter a name first")
			return

		if state.current_frame is None:
			FaceRecognitionController.set_status(state, "No frame available")
			return

		image_path = FaceRecognitionController.save_current_frame(state, name)

		FaceRecognitionController.set_status(state, "Registering " + name + "...")

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

		FaceRecognitionController.set_status(state,"Registered " + str(inserted) + " face record for " + name)

		print("Registered image:", image_path)

	##SAVES REGISTERED IMAGE ON COMPUTER AS IMAGE FOR TESTING

	@staticmethod
	def save_current_frame(state: FaceRecognitionState, name: str) -> Path:
		person_folder = DB_PATH / name
		person_folder.mkdir(parents=True, exist_ok=True)

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		image_path = person_folder / (timestamp + ".jpg")

		cv.imwrite(str(image_path), state.current_frame) # type: ignore

		return image_path

	#HELPER FUNCTION, CALL THIS WHEN CHANGING THE STATUS IN THE UI WINDOW

	@staticmethod
	def set_status(state: FaceRecognitionState, text: str) -> None:
		state.status_message = text
		state.status_text.set("Status: " + text)
		state.window.update_idletasks()

	@staticmethod #DOES NOTHING FOR NOW, FOR LIVENESS DETECTION
	def check_liveness(state: FaceRecognitionState) -> bool | None: #RETURNS TRUE OR FALSE
		if not state.liveness_enabled:
			state.liveness_passed = False
			return None


		state.liveness_passed = True
		return True

	@staticmethod #DOES NOTHING FOR NOW, FOR EMOTION DETECTION
	def detect_emotion(state: FaceRecognitionState) -> str | None:
		if not state.emotion_enabled: #IF NOT TRUE
			state.last_emotion = None #RETURN NOTHING
			return None

		state.last_emotion = None #IF SOMEHOW TRUE, RETURN NOTHING
		return None
