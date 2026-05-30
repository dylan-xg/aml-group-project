from pathlib import Path as _Path

import cv2 as _cv2
from deepface import DeepFace as _DeepFace

from panopticon.settings import SETTINGS as _SETTINGS
from panopticon.typing import Frame as _Frame


# Temporary and for reference only, proper implementation will be in a different module
def register(name: str, frames: list[_Frame]):
	for i, f in enumerate(frames):
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

	def consider_new_face(self, frame: _Frame, distance: float):
		"""If the distance is acceptable, save the frame."""
		_SETTINGS.REGISTRATION_THRESHOLD_MIN
		_SETTINGS.REGISTRATION_THRESHOLD_MAX
		_SETTINGS.NUM_REGISTRATION_IMAGES

	def finalise_registration(self):
		if _SETTINGS.USE_POSTGRES_DB:
			# TODO Will call an external function simiar to above.
			register(self.name, self.images)
			pass
		else:
			save_dir = _Path(_SETTINGS.LOCAL_DATABASE_PATH) / self.name
			save_dir.mkdir(parents=True, exist_ok=True)
			for i, frame in enumerate(self.images):
				file_path = save_dir / f"{self.name}_{i}.png"
				_cv2.imwrite(str(file_path), frame)
