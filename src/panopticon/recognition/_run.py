"""Execution of face recognition."""

from typing import Any as _Any, TypeAlias, cast as _cast

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


# I would like to having this in typing but it causes a circular dependency
# that I am too tired to try and fix.
FaceDistancePair: TypeAlias = tuple[_Face, float]
CategorisedFaces: TypeAlias = tuple[list[FaceDistancePair], list[FaceDistancePair]]


class FaceRecognitionSystem:
	"""Encapsulates face detection, state, and registration logic."""

	# The existence of this determines registration mode.
	_registration_face: _NewFace | None = None

	@classmethod
	def _get_faces(cls, frame_crop: _Frame, /) -> list[_pd.DataFrame] | None:
		"""Call deepface with the tight frame crop and standardise the result.
		Maintains nested list detection and critical exception reporting.
		"""
		try:
			if _SETTINGS.USE_POSTGRES_DB:
				detected_faces: list[_pd.DataFrame] = _DeepFace.search(
					img=frame_crop,
					model_name=_SETTINGS.MODEL_NAME,
					detector_backend=_SETTINGS.DETECTOR_BACKEND,
					distance_metric=_SETTINGS.DISTANCE_METRIC,
					enforce_detection=False,
					normalization=_SETTINGS.MODEL_NAME,
					k=_SETTINGS.TOP_K,
				)
			else:
				dfs: list[_pd.DataFrame] | list[list[dict[str, _Any]]] = _DeepFace.find(
					img_path=frame_crop,
					db_path=str(object=_SETTINGS.LOCAL_DATABASE_PATH),
					model_name=_SETTINGS.MODEL_NAME,
					distance_metric=_SETTINGS.DISTANCE_METRIC,
					enforce_detection=False,
					detector_backend=_SETTINGS.DETECTOR_BACKEND,
					k=_SETTINGS.TOP_K,
					normalization=_SETTINGS.MODEL_NAME,
					silent=True,
				)

				if len(dfs) > 0 and isinstance(dfs[0], list):
					raise RuntimeError("Doing batched for some reason.")

				detected_faces = _cast(list[_pd.DataFrame], dfs)

			if len(detected_faces) > 0:
				return detected_faces

			return None

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
			raise
		except Exception as e:
			print(f"Unexpected {type(e)}: {e}")
			return None

	@classmethod
	def _parse_face_crop(cls, face_crop: _Frame, box: _Box) -> FaceDistancePair:
		"""Process an individual face crop against the database."""
		dfs: list[_pd.DataFrame] | None = cls._get_faces(face_crop)

		distance: float = float("inf")
		identity_label: str = "Unknown"

		if dfs is not None and len(dfs) > 0:
			df: _pd.DataFrame = dfs[0]
			if not df.empty:
				best_match: _pd.Series[_Any] = df.iloc[0]
				distance = float(best_match["distance"])

				if distance <= _SETTINGS.UNKNOWN_THRESHOLD:
					if _SETTINGS.USE_POSTGRES_DB:
						identity_label = str(best_match["img_name"])
					else:
						identity_label = str(best_match["identity"])

		scale: float = 0.5 * _SETTINGS.TEXT_SCALE
		identity: _Text = _Text(label=identity_label, scale=scale)
		face: _Face = _Face(image=face_crop, box=box, texts=[identity])

		return (face, distance)

	@classmethod
	def _extract_faces(cls, frame: _Frame) -> CategorisedFaces:
		"""Isolate bounding boxes and classify them as known or unknown."""
		known_faces: list[FaceDistancePair] = []
		unknown_faces: list[FaceDistancePair] = []

		try:
			extracted: list[dict[str, _Any]] | list[list[dict[str, _Any]]] = (
				_DeepFace.extract_faces(
					img_path=frame,
					detector_backend=_SETTINGS.DETECTOR_BACKEND,
					enforce_detection=True,
					align=True,
				)
			)

			if len(extracted) > 0 and isinstance(extracted[0], list):
				raise RuntimeError("Doing batched for some reason.")

			extracted = _cast(list[dict[str, _Any]], extracted)
		except ValueError:
			# Raised when enforce_detection is True and no faces are located.
			extracted = []

		for face_obj in extracted:
			area: dict[str, int] = face_obj["facial_area"]
			left: int = int(area["x"])
			top: int = int(area["y"])
			width: int = int(area["w"])
			height: int = int(area["h"])

			right: int = left + width
			bottom: int = top + height

			box: _Box = _Box(left=left, top=top, right=right, bottom=bottom)
			face_crop: _Frame = frame[top:bottom, left:right]

			if face_crop.size == 0:
				continue

			face: _Face
			distance: float
			face, distance = cls._parse_face_crop(face_crop=face_crop, box=box)

			if face.texts[0].label == "Unknown":
				unknown_faces.append((face, distance))
			else:
				known_faces.append((face, distance))

		return known_faces, unknown_faces

	@classmethod
	def _handle_registration(
		cls,
		faces: CategorisedFaces,
		drawer: _Drawer,
		frame_shape: tuple[int, ...],
	) -> None:
		"""Process the registration logic if registration mode is active."""
		if cls._registration_face is None:
			return

		known_faces: list[FaceDistancePair]
		unknown_faces: list[FaceDistancePair]
		known_faces, unknown_faces = faces
		scale: float = 1 * _SETTINGS.TEXT_SCALE

		if len(known_faces) > 1 and len(unknown_faces) > 1:
			drawer.texts = _Text(
				label="Registration: More than once face detected",
				scale=scale,
				position=(5, frame_shape[0] - 10),
			)
			return

		face: _Face
		distance: float
		face, distance = (unknown_faces + known_faces)[0]
		is_complete: bool = cls._registration_face.consider_new_face(
			frame=face.image, distance=distance
		)

		if is_complete:
			msg: str = f"Registration Complete: {cls._registration_face.name}"
			cls._registration_face = None
		else:
			msg = (
				f"Registering {cls._registration_face.name} "
				f"({cls._registration_face.count}/{_SETTINGS.NUM_REGISTRATION_IMAGES})"
			)

		drawer.texts = _Text(label=msg, scale=scale, position=(5, frame_shape[0] - 10))

	@classmethod
	def _process_faces(cls, frame: _Frame, drawer: _Drawer) -> None:
		"""Extract faces, call modules, do registration stuff."""
		faces: CategorisedFaces = cls._extract_faces(frame=frame)
		known_faces: list[FaceDistancePair]
		unknown_faces: list[FaceDistancePair]
		known_faces, unknown_faces = faces

		if len(known_faces) == 0 and len(unknown_faces) == 0:
			if cls._registration_face is not None:
				msg = "Registration: No face detected"
			else:
				msg = "No faces detected"

			scale: float = 0.5 * _SETTINGS.TEXT_SCALE
			drawer.texts = _Text(
				label=msg, scale=scale, position=(5, frame.shape[0] - 10)
			)
			return

		for face, _ in known_faces + unknown_faces:
			drawer.faces = face

		_run_enabled_modules(drawer)

		if cls._registration_face is not None:
			cls._handle_registration(
				faces=faces, drawer=drawer, frame_shape=frame.shape
			)

	@classmethod
	def detect_in_frame(cls, frame: _Frame, /) -> _Frame:
		"""Callback for each frame. Returns a frame that has been drawn on."""
		drawer: _Drawer = _Drawer()
		cls._process_faces(frame=frame, drawer=drawer)
		return drawer.draw_onto_frame(frame)

	@classmethod
	def enable_registration_mode(cls, name: str, /) -> None:
		"""Start registration of a new face."""
		cls._registration_face = _NewFace(name)
