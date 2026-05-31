"""This is our recognition model."""

from abc import ABC as _ABC, abstractmethod as _abstractmethod
from pathlib import Path as _Path

import keras as _keras
import tensorflow as _tf
from keras.src import Functional as _Functional


class ExampleModel:
	"""For testing."""

	def __init__(self, name: str, input_shape: tuple[int, int, int]) -> None:
		self.name = name

		# self.input_shape = self.embedding_model.input_shape[1:3]
		# self.output_shape = self.embedding_model.output_shape[-1]

		# self.IMG_SIZE = self.input_shape
		# self.IMG_LENGTH = self.IMG_SIZE[0]
		# self.IMG_SHAPE = self.embedding_model.input_shape[1:]

		self.normalization = self.name
		# example values
		self.thresholds = {
			"cosine": 0.4,
			"euclidean": 0.9,
			"euclidean_l2": 0.9,
			"angular": 0.3,
		}

		self.confidences = {}

		# preprocess_layer = _keras.applications.mobilenet_v2.preprocess_input
		# weights = _keras.applications.MobileNetV2(
		# include_top=False,
		# input_shape=input_shape
		# )
		weights: _Functional = _keras.applications.EfficientNetV2B2(
			include_top=False, input_shape=input_shape
		)
		weights.trainable = False
		globalavg_layer = _keras.layers.GlobalAveragePooling2D()

		inputs = _keras.Input(shape=input_shape)
		# x: _Any = preprocess_layer(inputs)
		x = weights(inputs, training=False)
		latent_dim = globalavg_layer(x)
		self.embedding_model: _Functional = _Functional(
			inputs=inputs, outputs=latent_dim, trainable=False
		)

	def normalize(self, img):
		"""Normalise input image before embedding."""

		# For now, do nothing.
		# Later, replace this with the preprocessing used during training.
		return img


@_keras.saving.register_keras_serializable()
class L2Normalize(_keras.layers.Layer):
	def call(self, inputs):
		return _tf.math.l2_normalize(inputs, axis=1)

	def get_config(self):
		config = super().get_config()
		return config


class BaseDeepFaceEmbeddingModel(_ABC):
	"""Base wrapper for any custom embedding model we want DeepFace to use."""

	def __init__(
		self,
		name: str,
		model_path: str | _Path,
		thresholds: dict[str, float],
		confidences: dict[str, dict[str, dict[str, float]]] | None = None,
	) -> None:
		self.name = name

		if isinstance(model_path, str):
			model_path = _Path(model_path)

		self.embedding_model: _Functional = _keras.models.load_model(  # type: ignore
			model_path,
			compile=False,
			safe_mode=False,
			custom_objects={
				"L2Normalize": L2Normalize,
			},
		)

		self.input_shape = self.embedding_model.input_shape[1:3]
		self.output_shape = self.embedding_model.output_shape[-1]

		self.IMG_SIZE = self.input_shape
		self.IMG_LENGTH = self.IMG_SIZE[0]
		self.IMG_SHAPE = self.embedding_model.input_shape[1:]

		self.thresholds = thresholds
		self.confidences = confidences or {}

		self.normalization = self.name

	@_abstractmethod
	def normalize(self, img):
		"""Normalise input image before embedding."""
		pass


class CustomClassifierEmbeddingModel(BaseDeepFaceEmbeddingModel):
	def __init__(
		self,
		name: str,
		model_path: str | _Path,
	) -> None:
		thresholds: dict[str, float] = {
			"cosine": 0.4894155263900757,
			"euclidean": 0.9893589019775391,
			"euclidean_l2": 0.9893589615821838,
			"angular": 0.3294290602207184,
		}

		confidences: dict[str, dict[str, dict[str, float]]] = {
			name: {
				"cosine": {
					"w": -8.771574193202731,
					"b": 5.000415759855407,
					"normalizer": 0.8635047674179077,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0,
				},
				"euclidean": {
					"w": -12.192513030616036,
					"b": 9.148279809318405,
					"normalizer": 1.3141573667526245,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0,
				},
				"euclidean_l2": {
					"w": -12.192511846803123,
					"b": 9.14827915775775,
					"normalizer": 1.3141573667526245,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0,
				},
				"angular": {
					"w": -11.727093289922996,
					"b": 8.451637254382216,
					"normalizer": 0.45641618967056274,
					"denorm_max_true": 99.0,
					"denorm_min_true": 50.0,
					"denorm_max_false": 49.0,
					"denorm_min_false": 0.0,
				},
			}
		}

		super().__init__(
			name=name,
			model_path=model_path,
			thresholds=thresholds,
			confidences=confidences,
		)

	def normalize(self, img):
		# Original custom classifier behaviour.
		# Preprocessing was left to the model itself.
		return img


class MetricLearningEmbeddingModel(BaseDeepFaceEmbeddingModel):
	def __init__(
		self,
		name: str,
		model_path: str | _Path,
	) -> None:
		thresholds: dict[str, float] = {
			"cosine": 0.40,
			"euclidean": 1.00,
			"euclidean_l2": 1.00,
			"angular": 0.33,
		}

		super().__init__(
			name=name,
			model_path=model_path,
			thresholds=thresholds,
			confidences={},
		)

	def normalize(self, img):
		img = img.astype("float32")

		if img.max() > 1.0:
			img = img / 255.0

		return img
