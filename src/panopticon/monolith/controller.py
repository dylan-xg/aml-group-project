
from datetime import datetime
from pathlib import Path
import tkinter as tk
import time

import cv2 as cv
from deepface import DeepFace
import pandas as pd

from .. import video_feed as VideoFeed
from .settings import *
from ..typing import Frame


#IF VIDEO DOES NOT DISPLAY SET recognition_enabled TO FALSE
class FaceRecognitionController:
	def __init__(self) -> None:
		self.window = tk.Tk()
		self.window.title(string='Face Recognition Control')
		self.window.geometry(newGeometry='500x170')

		self.name_label = tk.Label(master=self.window, text='Name:')
		self.name_label.grid(row=0, column=0, padx=10, pady=10)

		self.name_entry = tk.Entry(master=self.window, width=30)
		self.name_entry.grid(row=0, column=1, padx=10, pady=10)

		self.status_text = tk.StringVar()
		self.status_text.set(value='Status: Ready')

		self.status_label = tk.Label(master=self.window, textvariable=self.status_text)
		self.status_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

		self.register_button = tk.Button(
			master=self.window,
			text='Register Unknown Face',
			command=lambda: self._register_unknown()
		)
		self.register_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

		self.current_frame: Frame | None = None

		self.recognition_enabled: bool = True #TO SEE ANY ISSUES DEEPFACE MIGHT BE CAUSING
		self.last_match_name: str | None = None
		self.last_match_confidence: float | None = None
		self.last_unknown: bool = False

		self.registration_required: bool = False

		self.liveness_enabled: bool = False
		self.liveness_passed: bool = True

		self.emotion_enabled: bool = False
		self.last_emotion: str | None = None

		self.status_message: str = 'Ready'

		self.last_recognition_time: float = 0.0
		self.recognition_interval: float = 1.5 #YOU MIGHT HAVE TO RAISE THIS IF CPU IS SLOW, VIDEO WILL NOT APPEAR


	def run(self, source: int | str | Path) -> None:
		try:
			VideoFeed.process_video(
				capture_location=source,
				callback=lambda frame: self._video_callback(frame),
				display_config=VideoFeed.DisplayConfig.OpenCV(frametime=1 / 15)
			)

		finally:
			print('Exiting Application...')
			try: self.window.destroy()
			except tk.TclError: pass


	def _video_callback(self, frame: Frame, /) -> Frame:
		self.current_frame = frame.copy()
		self._recognise_current_frame()
		try: self.window.update()
		except tk.TclError: raise KeyboardInterrupt
		return frame


	def _should_run_recognition(self) -> bool:
		if not self.recognition_enabled: return False
		if self.current_frame is None: return False
		current_time: float = time.time()
		if current_time - self.last_recognition_time < self.recognition_interval:
			return False

		self.last_recognition_time = current_time
		return True


	def _recognise_current_frame(self) -> None:
		if not self._should_run_recognition(): return
		start_time: float = time.time()
		dfs: list[pd.DataFrame] = DeepFace.search(
			img=self.current_frame, # type: ignore
			model_name=RECOGNITION_MODEL,
			distance_metric=DISTANCE_METRIC,
			normalization=RECOGNITION_MODEL,
			database_type='postgres',
			search_method='exact',
			enforce_detection=False
		)
		self._process_search_result(dfs)
		elapsed_time: float = time.time() - start_time
		print(f'Recognition time: {elapsed_time:.3f} seconds')


	def _process_search_result(self, dfs: list[pd.DataFrame], /) -> None:
		if len(dfs) == 0 or dfs[0].empty:
			self._handle_unknown()
			return

		best_match = dfs[0].iloc[0]
		best_match['distance']
		best_match['threshold']
		if 'img_name' in best_match.index:
			matched_name = str(best_match['img_name'])
		elif 'identity' in best_match.index:
			matched_name = str(best_match['identity'])
		else:
			matched_name = 'Unknown match'

		confidence = None
		if 'confidence' in best_match.index:
			confidence = float(best_match['confidence'])

		print(
			'Distance:', best_match['distance'],
			'Threshold:', best_match['threshold'],
			'Confidence:', best_match['confidence']
		)
		liveness_result: bool | None = self._check_liveness()
		emotion_result: str | None = self._detect_emotion()
		self._handle_match(
			matched_name=matched_name,
			confidence=confidence,
			liveness_result=liveness_result,
			emotion_result=emotion_result
		)


	def _handle_match(
		self,
		matched_name: str,
		confidence: float | None = None,
		liveness_result: bool | None = None,
		emotion_result: str | None = None
	) -> None:
		self.last_match_name = matched_name
		self.last_match_confidence = confidence
		self.last_unknown = False
		self.registration_required = False
		confidence_txt: str = f'({confidence:.2f}%)' if confidence is not None else ''
		liveness_text: str = {True: 'passed', False: 'failed', None: 'not checked'}[liveness_result]
		emotion_text: str = emotion_result if emotion_result is not None else 'not checked'
		self._set_status(
			f'Recognised: {matched_name} {confidence_txt} |'
			f'Liveness: {liveness_text} |'
			f'Emotion: {emotion_text}'
		)


	def _handle_unknown(self) -> None:
		self.last_match_name = None
		self.last_match_confidence = None
		self.last_unknown = True
		self.registration_required = True
		self._set_status('Unknown face - enter name and register')


	def _register_unknown(self) -> None:
		name: str = self.name_entry.get().strip()
		if name == '':
			self._set_status('Enter a name first')
			return

		if self.current_frame is None:
			self._set_status('No frame available')
			return

		image_path: Path = self._save_current_frame(name)
		self._set_status(f'Registering {name}...')
		result: dict = DeepFace.register(
			img=str(image_path),
			img_name=name,
			model_name=RECOGNITION_MODEL,
			normalization=RECOGNITION_MODEL,
			database_type='postgres',
			enforce_detection=False
		)
		inserted = result.get('inserted', 1)
		self.registration_required = False
		self.last_unknown = False
		self.last_match_name = name
		self._set_status(f'Registered {inserted} face record for {name}')
		print('Registered image:', image_path)


	##SAVES REGISTERED IMAGE ON COMPUTER AS IMAGE FOR TESTING
	def _save_current_frame(self, name: str, /) -> Path:
		person_folder: Path = DB_PATH / name
		person_folder.mkdir(parents=True, exist_ok=True)
		timestamp: str = datetime.now().strftime(format='%Y%m%d_%H%M%S')
		image_path: Path = person_folder / (timestamp + '.jpg')
		cv.imwrite(str(image_path), self.current_frame) # type: ignore
		return image_path


	#HELPER FUNCTION, CALL THIS WHEN CHANGING THE STATUS IN THE UI WINDOW
	def _set_status(self, text: str, /) -> None:
		self.status_message = text
		self.status_text.set('Status: ' + text)
		self.window.update_idletasks()


	#DOES NOTHING FOR NOW, FOR LIVENESS DETECTION
	def _check_liveness(self) -> bool | None:
		self.liveness_passed = self.liveness_enabled
		return self.liveness_enabled or None


	#DOES NOTHING FOR NOW, FOR EMOTION DETECTION
	def _detect_emotion(self) -> str | None:
		if not self.emotion_enabled:
			self.last_emotion = None
			return None

		self.last_emotion = None #IF SOMEHOW TRUE, RETURN NOTHING
		return None
