
from pathlib import Path as _Path
from typing import (
	Iterable as _Iterable,
	Literal as _Literal
)

import numpy as _np
import numpy.typing as _npt
from scipy.spatial.distance import cdist as _cdist
import tensorflow as _tf
from tensorflow import config as _tf_config # type: ignore
from tensorflow.data import AUTOTUNE as _TF_AUTOTUNE # type: ignore

from src.panopticon.recognition import ExampleModel
#from src.panopticon.recognition._distance import euclidean_distance

type ndarr = _npt.NDArray

def _config_gpu():
	GPUS: list[_tf_config.PhysicalDevice] = _tf_config.list_physical_devices(device_type='GPU')
	print(f'{len(GPUS)} GPU(s): {GPUS}')

	try:
		_tf_config.experimental.set_memory_growth(GPUS[0], True)
	except:
		# Invalid device or cannot modify virtual devices once initialized.
		pass


_config_gpu()

def compare_faces(
	faces: ndarr,
	model: ExampleModel,
	metric: _Literal['euclidean'] | _Literal['cosine']
) -> ndarr:
	"""Calculate the pairwise Euclidean distance matrix for face embeddings.

	Parameters
	----------
	faces : _npt.NDArray
		A batched array of images containing faces.
	model : ExampleModel or derived
		The model used to calculate the embeddings.

	Returns
	-------
	npt.NDArray
		A matrix of size faces^2 containing the euclidean distance between each face.

	Notes
	-----
		The image size of the faces should match the model input size.
	"""
	embeddings: ndarr = _np.asarray_chkfinite(
		model.embedding_model.predict_on_batch(faces)
	)
	# Scipy euclidean distance calculation, much better for arrays.
	return _cdist(embeddings, embeddings, metric=metric)


def compare_faces_from_path(
	faces: _Iterable[_Path],
	image_size: tuple[int, int],
	model: ExampleModel,
	metric: _Literal['euclidean'] | _Literal['cosine']
) -> ndarr:

	def _load_image(filepath: _tf.Tensor, /) -> _tf.Tensor:
		raw_image: _tf.Tensor = _tf.io.read_file(filepath)
		# Can also use _tf.image.decode_png
		image: _tf.Tensor = _tf.io.decode_png(raw_image, channels=3)
		image = _tf.image.resize(image, image_size)
		return image


	def _cast_image(image: _tf.Tensor, /) -> _tf.Tensor:
		return _tf.cast(x=image, dtype=_tf.float32)


	faces_list: list[str] = [str(f) for f in faces]
	total_faces: int = len(faces_list)
	dataset: _tf.data.Dataset = _tf.data.Dataset.from_tensor_slices(tensors=faces_list)
	# Chaining calls
	dataset = dataset.map(
		map_func=_load_image,
		num_parallel_calls=_TF_AUTOTUNE
	).batch(
		batch_size=total_faces
	).map(
		map_func=_cast_image,
		num_parallel_calls=_TF_AUTOTUNE
	).prefetch(
		buffer_size=_TF_AUTOTUNE
	)
	image_batch: npt.NDArray = dataset.as_numpy_iterator().next() # type: ignore
	return compare_faces(faces=image_batch, model=model, metric=metric)
