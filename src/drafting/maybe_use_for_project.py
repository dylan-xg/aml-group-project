
import os
from pathlib import Path
from typing import Any

import cv2 as cv
from deepface import DeepFace
from deepface.modules import verification
from deepface.modules.exceptions import FaceNotDetected
import pandas as pd

from ..panopticon import video_feed as VideoProcessor


# ==== Settings ====

FORCE_VID = True

PRERECORDING_PATH = Path('data/testing/recording.avi')

if 'SSH_CLIENT' in os.environ:
	print('Remote session detected. Using video file.')
	VIDEO_SOURCE = PRERECORDING_PATH
else:
	print('Local session detected. Using live webcam.')
	VIDEO_SOURCE = PRERECORDING_PATH if FORCE_VID else 0

RECOGNITION_MODEL = 'Facenet'
DISTANCE_METRIC = 'euclidean_l2'

FACE_YOUNGER_PATH = Path('data/testing/faces_db/Dylan/younger.jpg')
FACE_OLDER_PATH = Path('data/testing/faces_db/Dylan/peak.jpg')
FACE_RECENT_PATH = Path('data/testing/faces_db/Dylan/webcam_smirk.png')

# ==== Functions ====

def convert_path_to_string(path: Path, /) -> str:
	all_suffixes = ''.join(path.suffixes)
	base_name = path.name.removesuffix(all_suffixes)
	return '_'.join(path.parts[:-1] + (base_name,))


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
			normalization=model
		)
		sum += result['inserted']
	return sum


def embed_face(face, /) -> list[float]:
	represent_dict: list[dict[str, Any]] | list[list[dict[str, Any]]] = DeepFace.represent(
		img_path=face,
		model_name=RECOGNITION_MODEL,
		normalization=RECOGNITION_MODEL
	)

	# Match the type
	match represent_dict[0]:
		case list() as inner_list:
			return inner_list[0]['embedding']
		case dict() as inner_dict:
			return inner_dict['embedding']
		case _:
			raise TypeError("Unexpected data structure.")


def verify_against_image(frame: cv.typing.MatLike):
	try:
		# Will throw an exception if no face is found
		frame_embedding: list[float] = embed_face(frame)

		#result: dict = DeepFace.verify(
		#	img1_path=frame_embedding,
		#	img2_path=face_embedding,
		#	model_name=RECOGNITION_MODEL,
		#	distance_metric=DISTANCE_METRIC,
		#	normalization=RECOGNITION_MODEL
		#)
		#print(result)

		# Find the specific threshold for Facenet and euclidean_l2
		threshold = verification.find_threshold(RECOGNITION_MODEL, DISTANCE_METRIC)

		# Calculate the distance directly
		distance = verification.find_distance(frame_embedding, face_embedding, DISTANCE_METRIC)

		is_verified = distance <= threshold
		print({'verified': is_verified, 'distance': distance, 'threshold': threshold})

	except FaceNotDetected:
		print('No face found.')


# ==== Execution ====

DeepFace.build_model(RECOGNITION_MODEL)

result: dict = DeepFace.verify(
	img1_path=str(FACE_YOUNGER_PATH),
	img2_path=str(FACE_OLDER_PATH),
	model_name=RECOGNITION_MODEL,
	distance_metric=DISTANCE_METRIC,
	normalization=RECOGNITION_MODEL
)
print(result)

# Add faces to the database
register([FACE_YOUNGER_PATH, FACE_OLDER_PATH, FACE_RECENT_PATH])

DeepFace.build_index(RECOGNITION_MODEL)

dfs: list[pd.DataFrame] = DeepFace.search(
	img=str(FACE_RECENT_PATH),
	model_name=RECOGNITION_MODEL,
	normalization=RECOGNITION_MODEL
)
print(dfs)

face_embedding: list[float] = embed_face(str(FACE_RECENT_PATH))

VideoProcessor.process_video(
	VIDEO_SOURCE,
	callback=verify_against_image,
	display_config=VideoProcessor.DisplayConfig.OpenCV(frametime=1./10)
)
