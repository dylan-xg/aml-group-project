print("Hello World") #DOES PYTHON EVEN WORK?

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2 as cv
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from deepface import DeepFace
from deepface.modules import preprocessing
from deepface.modules.exceptions import FaceNotDetected
from deepface.modules import modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.models.facial_recognition.Facenet import (load_facenet128d_model) #LEGACY, for testing if the issue is your model file.
from deepface.modules.verification import thresholds
from deepface.modules.verification import confidences

from tensorflow.keras.models import load_model


import tkinter as tk
from datetime import datetime


# ============================================================
# Custom DeepFace Runtime Extension

CUSTOM_MODEL = "MyModel" #YOUR MODEL'S NAME HERE, could be anything

def load_my_custom_model():

    model = load_model("my_model.keras") #put your .keras model file here

    return model

class NewModelClient(FacialRecognition):

    def __init__(self) -> None:

        self.model = load_my_custom_model()

        self.model_name = CUSTOM_MODEL

        self.input_shape = self.model.input_shape[1:3]

        self.output_shape = self.model.output_shape[-1]

        type(self.model)


modeling.AVAILABLE_MODELS["facial_recognition"][CUSTOM_MODEL] = NewModelClient #adding your model name to the avaliable model list in the right category, facial recognition

thresholds[CUSTOM_MODEL] = thresholds["Facenet"] #just here because i cloned facenet to test the code, format: (variable) thresholds: dict[str, Any]

confidences[CUSTOM_MODEL] = confidences["Facenet"] #just here because i cloned facenet to test the code, format: (variable) confidences: dict[str, dict[str, dict[str, float]]]


original_normalize_input = preprocessing.normalize_input


def custom_normalize_input(img, normalization="base"):

    if normalization == CUSTOM_MODEL:

        # your custom normalization logic, change for each model
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

        #do anything you want within these comments ^
        return img

    return original_normalize_input(img=img,normalization=normalization)

preprocessing.normalize_input = custom_normalize_input

# ============================================================


PRERECORDING_PATH = Path('data/recording.avi')

if 'SSH_CLIENT' in os.environ:
	print('Remote session detected. Using video file.')
	VIDEO_SOURCE = PRERECORDING_PATH
else:
	print('Local session detected. Using live webcam.')
	VIDEO_SOURCE = 0

# Make sure to start the postgresql daemon
os.environ['DEEPFACE_POSTGRES_URI'] = 'postgresql://postgres:PASSWORD@localhost:5432/deepface' ##CHANGED LINE TO BE MY DB, WITH PASSWORD!!!

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


# Over-engineered to all hell
class VideoProcessor:

	@dataclass
	class DisplayConfig:
		pass

	@dataclass
	class DisplayOption_Headless(DisplayConfig):
		pass

	@dataclass
	class DisplayOption_Jupyter(DisplayConfig):
		image_widget: widgets.Image

	@dataclass
	class DisplayOption_OpenCV(DisplayConfig):
		pass

	#DisplayConfig: TypeAlias = DisplayOption_Headless | DisplayOption_Jupyter | DisplayOption_OpenCV

	FrameCallbackType: TypeAlias = Callable[[cv.typing.MatLike], cv.typing.MatLike | None]

	_END_LOOP = False
	_GO_AGAIN = True

	@staticmethod
	def _display_video(
		display_option: DisplayConfig,
		frame: cv.typing.MatLike
	) -> bool:
		"""Will attempt to display an output based on the display option."""

		match display_option:
			case VideoProcessor.DisplayOption_Jupyter(image_widget=image_widget):
				ret: bool
				buffer: np.ndarray[Any, np.dtype[np.uint8]]
				ret, buffer = cv.imencode(ext='.jpg', img=frame)

				if not ret:
					print("Can't encode frame as image. Exiting ...")
					return VideoProcessor._END_LOOP

				image_widget.value = buffer.tobytes()

			case VideoProcessor.DisplayOption_OpenCV:
				#print('Frame should be displayed') ##THIS IS SPAMMING THE TERMINAL
				cv.imshow(winname='frame', mat=frame)

		return VideoProcessor._GO_AGAIN

	@staticmethod
	def _frame_loop(
		vid_cap: cv.VideoCapture,
		callback: FrameCallbackType,
		display_option: DisplayConfig,
		frametime: float
	) -> bool:
		"""This function is called continuously until either the video ends, or is interrupted."""

		start_time: float = time.time()

		ret: bool
		frame: cv.typing.MatLike
		ret, frame = vid_cap.read()

		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			return VideoProcessor._END_LOOP

		frame_result: cv.typing.MatLike | None = callback(frame)
		if frame_result is None: frame_result = frame

		if not VideoProcessor._display_video(
			display_option=display_option,
			frame=frame_result
		): return VideoProcessor._END_LOOP

		match display_option:
			case VideoProcessor.DisplayOption_OpenCV:
				if cv.waitKey(delay=int(1/frametime)) == ord('q'):
					return VideoProcessor._END_LOOP

			case _: # VideoProcessor.DisplayOption_Jupyter:
				# Enforce the framerate pacing
				elapsed_time: float = time.time() - start_time
				time_to_wait: float = frametime - elapsed_time
				if time_to_wait > 0: time.sleep(time_to_wait)

		return VideoProcessor._GO_AGAIN

	@staticmethod
	def process_video(
		capture_location: int | str | Path,
		/,
		callback: FrameCallbackType = lambda x: x,
		*,
		display_config: DisplayConfig | None = None,
		frametime: float = 1.0 / 30.0
	) -> None:
		"""Read from a video input and apply the callback to it.

		Args:
			capture_location (int | str | Path): What the video source is.
				- For a webcam input, use an int.
				- For a video file, use a str or Path.
			callback (FrameCallbackType: Callable[[cv.typing.MatLike], cv.typing.MatLike | None]): Function that will be called with each frame.
			frametime (float): In seconds, how long between each frame. Use 1 / fps if you want to pass in a framerate.
			display_type (DisplayType): Whether to output the frames, and if so, how should it be shown. Options:
				- Jupyter Notebook.
				- Qt window via OpenCV.

		Returns:
			None:
		"""

		# Null sentinel
		if display_config is None:
			display_config = VideoProcessor.DisplayOption_Jupyter(
				# JPEG is faster than PNG
				image_widget=widgets.Image(format='jpeg')
			)

		vid_cap = cv.VideoCapture(capture_location)

		if not vid_cap.isOpened():
			print('Cannot open video source.')
			return

		if isinstance(display_config, VideoProcessor.DisplayOption_Jupyter):
			display(display_config.image_widget)

		try:
			while VideoProcessor._frame_loop(
				vid_cap=vid_cap,
				callback=callback,
				display_option=display_config,
				frametime=frametime
			): pass

		except KeyboardInterrupt:
			print('Video stream interrupted.')

		finally:
			vid_cap.release()
                  

DeepFace.build_model(RECOGNITION_MODEL)


def convert_path_to_string(path: Path, /) -> str:
	# There is a strong temptation to turn this into a single statement, but it would compromise readability.
	all_suffixes = ''.join(path.suffixes)
	base_name = path.name.removesuffix(all_suffixes)
	return '_'.join(path.parts[:-1] + (base_name,))

	# Alternative that does looping stuff
	#while path.suffix:
	#	path = path.with_suffix('')
	#return '_'.join(path.parts)
      


def register(
	path: Path | list[Path],
	/,
	model: str = RECOGNITION_MODEL
) -> int:
	if not isinstance(path, list):
		path = [path]

	sum = 0

	for p in path:
		result: dict[str, int] = DeepFace.register(
			img=str(p),
			img_name=convert_path_to_string(p),
			model_name=model,
			normalization=model,
            enforce_detection=False #changed
		)
		sum += result['inserted']
    

	return sum


DeepFace.build_index(RECOGNITION_MODEL) 

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

class FaceRecognitionController:


    @staticmethod
    def run(VIDEO_SOURCE: int | str | Path) -> None:
        state = FaceRecognitionController.create_state()
        try:
            VideoProcessor.process_video(
                VIDEO_SOURCE,
                callback=lambda frame: FaceRecognitionController.video_callback(state, frame),
                frametime=1 / 15,
                display_config=VideoProcessor.DisplayOption_OpenCV
            )

        finally:
            print("Exiting Application...")
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

        current_time = time.time()

        if current_time - state.last_recognition_time < state.recognition_interval:
            return False

        state.last_recognition_time = current_time

        return True
    

    @staticmethod
    def recognise_current_frame(state: FaceRecognitionState) -> None:
        if not FaceRecognitionController.should_run_recognition(state):
            return

        start_time = time.time()

        dfs = DeepFace.search(
            img=state.current_frame,
            model_name=RECOGNITION_MODEL,
            distance_metric=DISTANCE_METRIC,
            normalization=RECOGNITION_MODEL,
            database_type="postgres",
            search_method="exact",
            enforce_detection=False
        )

        FaceRecognitionController.process_search_result(state, dfs)

        elapsed_time = time.time() - start_time

        print(f"Recognition time: {elapsed_time:.3f} seconds")


    @staticmethod
    def process_search_result(state: FaceRecognitionState, dfs: list[pd.DataFrame]) -> None:
        if len(dfs) == 0 or dfs[0].empty:
            FaceRecognitionController.handle_unknown(state)
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

        cv.imwrite(str(image_path), state.current_frame)

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
    



print("This could take a minute, give it time...")
print("Use the Q key to exit the application.")
FaceRecognitionController.run(VIDEO_SOURCE)