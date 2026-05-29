"""This is our recognition model."""

from pathlib import Path as _Path

import keras as _keras
from keras.src import Functional as _Functional


class ExampleModel:
	"""For testing."""

	def __init__(self, name: str = '', model_path: str | _Path = '') -> None:
		self.name = name
		self.model_path = _Path(model_path)

		self.embedding_model: _Functional = _keras.models.load_model( # type: ignore
			model_path,
			compile=False
		)


		self.input_shape = self.embedding_model.input_shape[1:3]
		self.output_shape = self.embedding_model.output_shape[-1]

		self.IMG_SIZE = self.input_shape
		self.IMG_LENGTH = self.IMG_SIZE[0]
		self.IMG_SHAPE = self.embedding_model.input_shape[1:]

		self.normalization = self.name
		#example values
		self.thresholds = {
			"cosine": 0.4,
			"euclidean": 0.9,
			"euclidean_l2": 0.9,
			"angular": 0.3
		}

		self.confidences = {}

		#preprocess_layer = _keras.applications.mobilenet_v2.preprocess_input
		#weights = _keras.applications.MobileNetV2(
		#	include_top=False,
		#	input_shape=self.IMG_SHAPE
		#)
		weights: _Functional = _keras.applications.EfficientNetV2B2(
			include_top=False,
			input_shape=self.IMG_SHAPE
		)
		weights.trainable = False
		globalavg_layer = _keras.layers.GlobalAveragePooling2D()

		inputs = _keras.Input(shape=self.IMG_SHAPE)
		#x: _Any = preprocess_layer(inputs)
		x = weights(inputs, training=False)
		latent_dim = globalavg_layer(x)
		self.embedding_model: _Functional = _Functional(inputs=inputs, outputs=latent_dim, trainable=False)

	def normalize(self, img):
		"""Normalise input image before embedding."""

		# For now, do nothing.
		# Later, replace this with the preprocessing used during training.
		return img


class CustomClassifierEmbeddingModel:

	def __init__(self, name: str = "CustomClassifierEmbeddingModel", model_path: str | _Path = "src/panopticon/model_weights/final_face_embedding_model._keras") -> None:
		self.name = name
		self.model_path = _Path(model_path)

		self.embedding_model: _Functional = _keras.models.load_model(  # type: ignore
			model_path,
			compile=False
		)

		self.input_shape = self.embedding_model.input_shape[1:3]
		self.output_shape = self.embedding_model.output_shape[-1]

		self.IMG_SIZE = self.input_shape
		self.IMG_LENGTH = self.IMG_SIZE[0]
		self.IMG_SHAPE = self.embedding_model.input_shape[1:]

		self.thresholds: dict[str, float] = {
			"cosine": 0.4894155263900757,
			"euclidean": 0.9893589019775391,
			"euclidean_l2": 0.9893589615821838,
			"angular": 0.3294290602207184
		}

		self.confidences: dict[str, dict[str, dict[str, float]]] = {
			self.name: {
				"cosine": {
					"w": -8.771574193202731,
					"b": 5.000415759855407,
					"normalizer": 0.8635047674179077,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0
				},
				"euclidean": {
					"w": -12.192513030616036,
					"b": 9.148279809318405,
					"normalizer": 1.3141573667526245,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0
				},
				"euclidean_l2": {
					"w": -12.192511846803123,
					"b": 9.14827915775775,
					"normalizer": 1.3141573667526245,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0
				},
				"angular": {
					"w": -11.727093289922996,
					"b": 8.451637254382216,
					"normalizer": 0.45641618967056274,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0
				}
			}
		}

		self.normalization = self.name


	def normalize(self, img):
		#left the processing to the model, so not any need, although deepface will throw a fit otherwise so it's here anyways

		return img
