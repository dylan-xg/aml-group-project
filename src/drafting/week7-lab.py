
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import cv2 as cv
import pandas as pd
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected

from ..monolith import video_processor as VP


# ==== Settings ====

FORCE_VID = True

PRERECORDING_PATH = Path('data/testing/on_site.mp4')

if 'SSH_CLIENT' in os.environ:
	print('Remote session detected. Using video file.')
	VIDEO_SOURCE = PRERECORDING_PATH
else:
	print('Local session detected. Using live webcam.')
	VIDEO_SOURCE = PRERECORDING_PATH if FORCE_VID else 0

RECOGNITION_MODEL = 'Facenet'
DISTANCE_METRIC = 'euclidean_l2'
DETECTOR_BACKEND = 'ssd'

DB_PATH = Path('data/testing/faces_db')
TOP_K = 1
BOX_COLOUR = (0, 0, 255)
"""BGR format."""
TEXT_COLOUR = (50, 50, 230)
"""BGR format."""

PATIENCE = 10
"""How many frames before swapping state."""

CSV_PATH = Path('data/testing/log.csv')

# ==== Classes ====

@dataclass
class Person:
	present: bool = False
	timer: int = 0


# ==== Functions ====

def get_faces(frame: cv.typing.MatLike, /) -> list[pd.DataFrame] | None:
	try:
		dfs: list[pd.DataFrame] | list[list[dict[str, Any]]] = DeepFace.find(
			img_path=frame,
			db_path=str(object=DB_PATH),
			model_name=RECOGNITION_MODEL,
			distance_metric=DISTANCE_METRIC,
			detector_backend=DETECTOR_BACKEND,
			k=TOP_K,
			normalization=RECOGNITION_MODEL,
			silent=True
		)
	except FaceNotDetected:
		return None

	if dfs and isinstance(dfs[0], list):
		raise Exception('Doing batched for some reason')

	if not len(dfs):
		return None

	return cast(list[pd.DataFrame], dfs)


def extract_name(identity: str, /) -> str:
	id_path = Path(identity)
	return ''.join(id_path.parts[-2:-1])


def draw_onto_frame(
	frame: cv.typing.MatLike,
	face: pd.Series,
	name: str
) -> cv.typing.MatLike:
	left: int = face['source_x']
	top: int = face['source_y']
	right: int = face['source_w']
	bottom: int = face['source_h']

	# Rectangle
	frame = cv.rectangle(
		img=frame,
		rec=(left, top, right, bottom),
		color=BOX_COLOUR
	)

	# Text
	cv.putText(
		img=frame,
		text=name,
		org=(left, top),
		fontFace=cv.FONT_HERSHEY_SIMPLEX,
		fontScale=0.8,
		color=TEXT_COLOUR,
		thickness=2,
		lineType=cv.LINE_AA
	)

	return frame


def write_to_csv(name: str, event: bool) -> None:
	event_text: str = 'ENTER' if event else 'EXIT'
	current_time = str(object=datetime.now().strftime(format='%Y-%m-%d %H:%M:%S'))
	with open(file=CSV_PATH, mode='a') as csvfile:
		writer: csv.DictWriter[str] = csv.DictWriter(
			f=csvfile, fieldnames=['Name', 'Event', 'Time']
		)
		writer.writerow(
			rowdict={'Name': name, 'Event': event_text, 'Time': current_time}
		)


def update_log(present: list[str]) -> None:
	# Update known people
	for name, person in log.items():
		is_present: bool = name in present
		# Check state change
		if person.present != is_present:
			# Check current timer
			if person.timer == PATIENCE:
				person.timer = 0
				person.present = is_present
				write_to_csv(name=name, event=is_present)
			else:
				# Update timer
				person.timer += 1

	# Check for first appearance
	for name in set(present) - log.keys():
		if name not in log:
			log[name] = Person(present=True)
			write_to_csv(name=name, event=True)


def face_detection(frame: cv.typing.MatLike) -> cv.typing.MatLike | None:
	detected_this_frame: list[str] = []
	detected_faces: list[pd.DataFrame] | None = get_faces(frame)

	if detected_faces:
		# Only one for loop needed this way.
		for detected_face in detected_faces:
			# For some reason, you can get empty dataframes
			if detected_face.empty:
				continue

			matched_face: pd.Series[Any] = detected_face.iloc[0]

			name: str = extract_name(matched_face['identity'])

			detected_this_frame.append(name)

			frame = draw_onto_frame(
				frame=frame,
				face=matched_face,
				name=name
			)

	update_log(detected_this_frame)

	return frame


# ==== Execution ====

_ = DeepFace.build_model(model_name=RECOGNITION_MODEL)

# Clear the file
with open(file=CSV_PATH, mode='w') as csvfile:
	pass

# Name and info
log: dict[str, Person] = {}

VP.process_video(
	VIDEO_SOURCE,
	callback=face_detection,
	display_config=VP.DisplayConfig.OpenCV(frametime=1./60)
)
