"""Execution of face recognition."""

from typing import Any as _Any, cast as _cast

import pandas as _pd
from deepface import DeepFace as _DeepFace
from deepface.modules.exceptions import (
	DimensionMismatchError as _DimensionMismatchError,
	EmptyDatasource as _EmptyDatasource,
	ImgNotFound as _ImgNotFound,
	PathNotFound as _PathNotFound,
)

from panopticon.drawer import (
	Box as _Box,
	Drawer as _Drawer,
	Face as _Face,
	Text as _Text,
)
from panopticon.modules import run_enabled_modules as _run_enabled_modules
from panopticon.settings import SETTINGS as _SETTINGS
from panopticon.typing import Frame as _Frame

from ._registration import NewFace as _NewFace


# The existence of this determines registration mode.
registration_face: _NewFace | None


def _handle_face(dataframe: _pd.DataFrame, frame: _Frame):
	"""Create a Face object for the passed in data.

	This will be used for the modules, and drawing onto the frame.
	"""
	# TODO Instead of passing a full dataframe,
	# pass only the necessary info to construct a Face.

	# Getting the best result.
	matched_face: _pd.Series[_Any] = dataframe.iloc[0]

	def _read_dataframe(
		identity: str, left: str, top: str, width: str, height: str
	) -> tuple[_Text, int, int, int, int]:
		_identity: _Text = _Text(label=matched_face[identity])
		_left: int = matched_face[left]
		_top: int = matched_face[top]
		_right: int = matched_face[width] + _left
		_bottom: int = matched_face[height] + _top
		return (_identity, _left, _top, _right, _bottom)

	identity: _Text
	left: int
	top: int
	right: int
	bottom: int
	if _SETTINGS.USE_POSTGRES_DB:
		# DeepFace.search dataframe options.
		SEARCH_LABELS = ("img_name", "target_x", "target_y", "target_w", "target_h")
		identity, left, top, right, bottom = _read_dataframe(*SEARCH_LABELS)
	else:
		# DeepFace.find dataframe options.
		FIND_LABELS = ("identity", "source_x", "source_y", "source_w", "source_h")
		identity, left, top, right, bottom = _read_dataframe(*FIND_LABELS)

	# Get the similarity distance
	distance: float = matched_face["distance"]

	# If the match is too weak, mark it as Unknown
	if distance > _SETTINGS.UNKNOWN_THRESHOLD:
		identity = _Text(label="Unknown")

	face_box: _Box = _Box(left=left, top=top, right=right, bottom=bottom)
	cropped_frame: _Frame = frame[left:right, top:bottom]
	return _Face(image=cropped_frame, box=face_box, texts=[identity]), distance


def _process_dataframe(dataframes: list[_pd.DataFrame], /):
	"""Read in the list of dataframe, extract a representation for each face."""
	if len(dataframes) == 0:
		return

	for df in dataframes:
		# Maybe use df.groupby()?
		best_match: _pd.Series[_Any] = df.iloc[0]
		known_faces = best_match[best_match["distance"] > _SETTINGS.UNKNOWN_THRESHOLD]
		unknown_faces = best_match[
			best_match["distance"] <= _SETTINGS.UNKNOWN_THRESHOLD
		]

	# Create a representation for all faces, tagging them as either known or unknown.


def _process_faces(frame: _Frame, drawer: _Drawer, faces: list[_pd.DataFrame] | None):
	"""idk this is basically just an inner function to allow dropout returns
	that still provoke the drawer."""
	global registration_face

	if faces is None:
		if registration_face is not None:
			msg = "Registration: No face detected"
		else:
			msg = "No faces detected"
		drawer.texts = _Text(label=msg, scale=2, position=(5, frame.shape[:1][0] - 10))
		return

	# The result of this should be used to improve the above and below code.
	_process_dataframe(faces)

	unknown_faces: list[tuple[_Face, float]] = []
	for detected_face in faces:
		if detected_face.empty:
			print("Empty")
			continue

		face, dist = _handle_face(dataframe=detected_face, frame=frame)

		if registration_face is not None:
			if face.get_name() == "Unknown":
				unknown_faces.append((face, dist))

		drawer.faces = face

	_run_enabled_modules(drawer=drawer)

	if len(unknown_faces) == 0:
		drawer.texts = _Text(
			label="Registration: No unknown face detected",
			scale=2,
			position=(5, frame.shape[0] - 10),
		)

	if len(unknown_faces) > 1:
		drawer.texts = _Text(
			label="Registration: Multiple unknown faces detected",
			scale=2,
			position=(5, frame.shape[0] - 10),
		)

	if registration_face is not None:
		registration_face.consider_new_face(
			unknown_faces[0][0].image, unknown_faces[0][1]
		)

		drawer.texts = _Text(
			label=(
				f"Registering {registration_face.name} "
				f"({len(registration_face.images)}/{_SETTINGS.NUM_REGISTRATION_IMAGES})"
			),
			scale=2,
			position=(5, frame.shape[0] - 10),
		)


def _get_faces(frame: _Frame, /) -> list[_pd.DataFrame] | None:
	"""Call deepface with the frame and standardise the result."""
	try:
		if _SETTINGS.USE_POSTGRES_DB:
			detected_faces: list[_pd.DataFrame] = _DeepFace.search(
				img=frame,
				model_name=_SETTINGS.MODEL_NAME,
				detector_backend=_SETTINGS.DETECTOR_BACKEND,
				distance_metric=_SETTINGS.DISTANCE_METRIC,
				enforce_detection=False,
				normalization=_SETTINGS.MODEL_NAME,
				k=_SETTINGS.TOP_K,
			)
		else:
			dfs: list[_pd.DataFrame] | list[list[dict[str, _Any]]] = _DeepFace.find(
				img_path=frame,
				db_path=str(object=_SETTINGS.LOCAL_DATABASE_PATH),
				model_name=_SETTINGS.MODEL_NAME,
				distance_metric=_SETTINGS.DISTANCE_METRIC,
				enforce_detection=False,
				detector_backend=_SETTINGS.DETECTOR_BACKEND,
				k=_SETTINGS.TOP_K,
				normalization=_SETTINGS.MODEL_NAME,
				silent=True,
			)

			if dfs and isinstance(dfs[0], list):
				raise Exception("Doing batched for some reason")

			detected_faces = _cast(list[_pd.DataFrame], dfs)

		return detected_faces if len(detected_faces) > 0 else None

	except _EmptyDatasource:
		# No images in local database, we don't care about this.
		return None
	except _DimensionMismatchError as e:
		print(f"Frame dimensions incorrect: {e}")
		quit()
	except _ImgNotFound as e:
		print(f"Error reading frame: {e}")
		quit()
	except _PathNotFound as e:
		print(f"Error reading path: {e}")
		quit()
	except ValueError as e:
		# No embeddings in postgres database, we don't care about this.
		if any(
			msg in str(e.args[0])
			for msg in [
				"No embeddings found in the database for the criteria",
				"You must call register some embeddings "
				"to the database before using search.",
			]
		):
			return None
		else:
			# Pass it down if it is a different exception.
			raise
	except Exception as e:
		print(f"Unexpected {type(e)}: {e}")


def detect_in_frame(frame: _Frame) -> _Frame:
	"""Callback for each frame.
	Returns a frame that has been drawn on.
	"""
	faces: list[_pd.DataFrame] | None = _get_faces(frame)
	drawer: _Drawer = _Drawer()
	_process_faces(frame=frame, drawer=drawer, faces=faces)
	return drawer.draw_onto_frame(frame)


def enable_registration_mode(name: str, /) -> None:
	"""Callback to start registration of a new face."""
	global registration_face
	registration_face = _NewFace(name)
