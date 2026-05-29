from pathlib import Path
from typing import Any, Iterable, Literal

import cv2 as cv
import pandas as pd
from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected

from src.panopticon.settings import SETTINGS
from src.panopticon.typing import Frame
from src.panopticon.ui import UserInterface
from src.panopticon.video_feed import VideoFeed


TESTING_VID = Path("data/testing/on_site.mp4")
# TESTING_VID = Path('data/testing/recording.avi')

DB_PATH = Path("data/testing/faces_db")
FACES = (
	Path("data/testing/faces_db/Dylan/younger.jpg"),
	Path("data/testing/faces_db/Dylan/peak.jpg"),
	Path("data/testing/faces_db/Dylan/webcam_smirk.png"),
	Path("data/testing/faces_db/Grace/grace_1.png"),
	Path("data/testing/faces_db/Grace/grace_2.png"),
)

RECOGNITION_MODEL = "Facenet"
DISTANCE_METRIC = "euclidean_l2"
DETECTOR_BACKEND = "ssd"
SEARCH_METHOD: Literal["exact"] | Literal["ann"] = "exact"

TOP_K = 1
BOX_COLOUR = (0, 0, 255)
"""BGR format."""
TEXT_COLOUR = (50, 50, 230)
"""BGR format."""


def convert_path_to_string(path: Path, /) -> str:
	all_suffixes = "".join(path.suffixes)
	base_name = path.name.removesuffix(all_suffixes)
	return "_".join(path.parts[:-1] + (base_name,))


def register(path: Path | Iterable[Path], /, model: str = RECOGNITION_MODEL) -> int:
	if isinstance(path, Path):
		path = (path,)

	sum = 0

	for p in path:
		result: dict[str, int] = DeepFace.register(
			img=str(p),
			img_name=convert_path_to_string(p),
			model_name=model,
			detector_backend=DETECTOR_BACKEND,
			normalization=model,
		)
		sum += result["inserted"]
	return sum


def extract_name(identity: str, /) -> str:
	return "".join(identity.split("_")[-2:-1])


def draw_onto_frame(
	frame: Frame, left: int, top: int, right: int, bottom: int, name: str | None = None
) -> Frame:
	# Rectangle
	frame = cv.rectangle(img=frame, rec=(left, top, right, bottom), color=BOX_COLOUR)

	# Text
	if name is not None:
		cv.putText(
			img=frame,
			text=name,
			org=(left, top),
			fontFace=cv.FONT_HERSHEY_SIMPLEX,
			fontScale=0.8,
			color=TEXT_COLOUR,
			thickness=2,
			lineType=cv.LINE_AA,
		)

	return frame


def face_detection_db(frame: Frame) -> Frame | None:
	try:
		detected_faces: list[pd.DataFrame] = DeepFace.search(
			img=frame,
			model_name=RECOGNITION_MODEL,
			detector_backend=DETECTOR_BACKEND,
			distance_metric=DISTANCE_METRIC,
			normalization=RECOGNITION_MODEL,
			k=TOP_K,
		)

		# Only one for loop needed this way.
		for detected_face in detected_faces:
			# For some reason, you can get empty dataframes
			if detected_face.empty:
				raise FaceNotDetected

			matched_face: pd.Series[Any] = detected_face.iloc[0]

			name: str = extract_name(matched_face["img_name"])

			frame = draw_onto_frame(
				frame=frame,
				left=matched_face["target_x"],
				top=matched_face["target_y"],
				right=matched_face["target_w"],
				bottom=matched_face["target_h"],
				name=name,
			)

	except FaceNotDetected:
		cv.putText(
			img=frame,
			text="NO FACE FOUND",
			org=(5, 50),
			fontFace=cv.FONT_HERSHEY_SIMPLEX,
			fontScale=1.2,
			color=TEXT_COLOUR,
			thickness=2,
			lineType=cv.LINE_AA,
		)

	except KeyError as e:
		print(e)

	finally:
		return frame


# Add faces to the database
register(FACES)

DeepFace.build_index(RECOGNITION_MODEL)

video_feed = VideoFeed(
	capture_location=TESTING_VID, callback=face_detection_db, frametime=1 / 60
)

app = UserInterface(
	width=SETTINGS.WINDOW_WIDTH,
	height=SETTINGS.WINDOW_HEIGHT,
	title="Test Window",
	video_feed=video_feed,
)

app.start()
