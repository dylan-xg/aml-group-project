from pathlib import Path as _Path

import cv2 as _cv2
from deepface import DeepFace as _DeepFace

from panopticon.settings import SETTINGS as _SETTINGS
from panopticon.typing import Frame as _Frame


# Temporary and for reference only, proper implementation will be in a different module
def register(name: str, frames: list[_Frame]) -> None:
	for i, f in enumerate(iterable=frames):
		_DeepFace.register(
			img=f,
			img_name=f"{name}_{i}",
			model_name=_SETTINGS.MODEL_NAME,
			detector_backend=_SETTINGS.DETECTOR_BACKEND,
			normalization=_SETTINGS.MODEL_NAME,
		)


class NewFace:
	"""This represents an unknown face undergoing registration."""

	def __init__(self, name: str, /) -> None:
		self.name: str = name
		self.images: list[_Frame] = []

	def consider_new_face(self, frame: _Frame, distance: float) -> bool:
		"""Evaluate the unknown face and store it if it meets requirements.

		Returns
		-------
		bool
			True if the required number of images has been collected, False otherwise.
		"""
		# Unrecognised due to an empty database will return infinity;
		# bypass bounds check to guarantee saving.
		if distance == float("inf") or (
			_SETTINGS.REGISTRATION_THRESHOLD_MIN
			<= distance
			<= _SETTINGS.REGISTRATION_THRESHOLD_MAX
		):
			self.images.append(frame)

		return len(self.images) >= _SETTINGS.NUM_REGISTRATION_IMAGES

	def finalise_registration(self) -> None:
		"""Save the collected images to the database or local directory."""
		if _SETTINGS.USE_POSTGRES_DB is True:
			register(name=self.name, frames=self.images)
		else:
			save_dir: _Path = _Path(_SETTINGS.LOCAL_DATABASE_PATH) / self.name
			save_dir.mkdir(parents=True, exist_ok=True)

			for i, frame in enumerate(self.images):
				file_path: _Path = save_dir / f"{self.name}_{i}.png"
				_cv2.imwrite(filename=str(file_path), img=frame)
