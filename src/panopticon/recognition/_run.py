"""Execution of face recognition."""

from typing import Any as _Any, cast as _cast

import pandas as _pd
from deepface import DeepFace as _DeepFace

from panopticon.drawer import (
	Box as _Box,
	Drawer as _Drawer,
	Face as _Face,
	Text as _Text,
)
from panopticon.modules import run_enabled_modules as _run_enabled_modules
from panopticon.settings import SETTINGS as _SETTINGS
from panopticon.typing import Frame as _Frame


def _handle_face(dataframe: _pd.DataFrame, frame: _Frame) -> _Face:
	UNKNOWN_THRESHOLD = 0.40

	# Getting the best result.
	matched_face: _pd.Series[_Any] = dataframe.iloc[0]

	# Get the similarity distance
	distance: float = matched_face["distance"]

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

	# If the match is too weak, mark it as Unknown
	if distance > UNKNOWN_THRESHOLD:
		identity = _Text("Unknown")

	face_box: _Box = _Box(left=left, top=top, right=right, bottom=bottom)
	cropped_frame: _Frame = frame[left:right, top:bottom]
	return _Face(image=cropped_frame, box=face_box, texts=[identity])


def _get_faces(frame: _Frame, /) -> list[_pd.DataFrame] | None:
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

	return detected_faces if len(detected_faces) > 1 else None


def detect_in_frame(frame: _Frame) -> _Frame:
	faces: list[_pd.DataFrame] | None = _get_faces(frame)

	if faces is None:
		return frame

	drawer: _Drawer = _Drawer()

	for detected_face in faces:
		if detected_face.empty:
			print("Empty")
			continue
		drawer.faces = _handle_face(dataframe=detected_face, frame=frame)

	_run_enabled_modules(drawer=drawer)

	return drawer.draw_onto_frame(frame)
