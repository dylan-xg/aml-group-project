
from pathlib import Path as _Path
from typing import Iterable as _Iterable

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist as _cdist
import tensorflow as tf
from tensorflow import config as tf_config # type: ignore
from tensorflow.data import AUTOTUNE as tf_AUTOTUNE # type: ignore

from src.panopticon.recognition import BaseModel
#from src.panopticon.recognition._distance import euclidean_distance

type ndarr = npt.NDArray

def _config_gpu():
	GPUS: list[tf_config.PhysicalDevice] = tf_config.list_physical_devices(device_type='GPU')
	print(f'{len(GPUS)} GPU(s): {GPUS}')

	try:
		tf_config.experimental.set_memory_growth(GPUS[0], True)
	except:
		# Invalid device or cannot modify virtual devices once initialized.
		pass


_config_gpu()

def compare_faces(
	faces: ndarr,
	model: BaseModel
) -> ndarr:
	"""Calculate the pairwise Euclidean distance matrix for face embeddings.

	Parameters
	----------
	faces : npt.NDArray
		A batched array of images containing faces.
	model : BaseModel or derived
		The model used to calculate the embeddings.

	Returns
	-------
	npt.NDArray
		A matrix of size faces^2 containing the euclidean distance between each face.

	Notes
	-----
		The image size of the faces should match the model input size.
	"""
	embeddings: ndarr = np.asarray_chkfinite(
		model.embedding_model.predict_on_batch(faces)
	)
	# Scipy euclidean distance calculation, much better for arrays.
	return _cdist(embeddings, embeddings, metric='euclidean')


def compare_faces_from_path(
	faces: _Iterable[_Path],
	image_size: tuple[int, int],
	model: BaseModel
) -> ndarr:

	def _load_image(filepath: tf.Tensor, /) -> tf.Tensor:
		raw_image: tf.Tensor = tf.io.read_file(filepath)
		# Can also use tf.image.decode_png
		image: tf.Tensor = tf.io.decode_png(raw_image, channels=3)
		image = tf.image.resize(image, image_size)
		return image


	def _cast_image(image: tf.Tensor, /) -> tf.Tensor:
		return tf.cast(x=image, dtype=tf.float32)


	faces_list: list[str] = [str(f) for f in faces]
	total_faces: int = len(faces_list)
	dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(tensors=faces_list)
	# Chaining calls
	dataset = dataset.map(
		map_func=_load_image,
		num_parallel_calls=tf_AUTOTUNE
	).batch(
		batch_size=total_faces
	).map(
		map_func=_cast_image,
		num_parallel_calls=tf_AUTOTUNE
	).prefetch(
		buffer_size=tf_AUTOTUNE
	)
	image_batch: npt.NDArray = dataset.as_numpy_iterator().next() # type: ignore
	return compare_faces(faces=image_batch, model=model)
