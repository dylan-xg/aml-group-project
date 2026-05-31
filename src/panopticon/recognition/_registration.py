from pathlib import Path as _Path

import cv2 as _cv2

from panopticon.settings import SETTINGS as _SETTINGS
from panopticon.typing import Frame as _Frame

from ._deepface import register as _register


class NewFace:
	"""This represents an unknown face undergoing registration."""

	def __init__(self, name: str, /) -> None:
		self.name: str = name
		self.count: int = 0

	def consider_new_face(self, frame: _Frame, distance: float) -> bool:
		"""Evaluate the unknown face and store it if it meets requirements.

		Returns
		-------
		bool
			True if the required number of images has been collected, False otherwise.
		"""
		# Handle the first image.
		if self.count == 0:
			self.count += 1
			self.register_to_database(frame)
			return False

		# Unrecognised due to an empty database will return infinity;
		# bypass bounds check to guarantee saving.
		print(distance)
		if distance == float("inf") or (
			_SETTINGS.REGISTRATION_THRESHOLD_MIN
			<= distance
			<= _SETTINGS.REGISTRATION_THRESHOLD_MAX
		):
			self.count += 1
			self.register_to_database(frame)

		return self.count >= _SETTINGS.NUM_REGISTRATION_IMAGES

	def register_to_database(self, frame: _Frame, /) -> None:
		"""Save the collected images to the database or local directory."""

		if _SETTINGS.USE_POSTGRES_DB is True:
			_register(name=self.name, frames=frame)
		else:
			save_dir: _Path = _Path(_SETTINGS.LOCAL_DATABASE_PATH) / self.name
			save_dir.mkdir(parents=True, exist_ok=True)
			file_path: _Path = save_dir / f"{self.name}_{self.count}.png"
			_cv2.imwrite(filename=str(file_path), img=frame)
