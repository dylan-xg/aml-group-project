"""This is our recognition model."""

import keras
from keras.src import Functional


class ExampleModel:
	"""For testing."""

	def __init__(self, name: str = '', image_length: int = 64) -> None:
		self.name = name
		self.IMG_LENGTH = image_length
		self.IMG_SIZE = (self.IMG_LENGTH, self.IMG_LENGTH)
		self.IMG_SHAPE = self.IMG_SIZE + (3,)

		#preprocess_layer = keras.applications.mobilenet_v2.preprocess_input
		#weights = keras.applications.MobileNetV2(
		#	include_top=False,
		#	input_shape=self.IMG_SHAPE
		#)
		weights: Functional = keras.applications.EfficientNetV2B2(
			include_top=False,
			input_shape=self.IMG_SHAPE
		)
		weights.trainable = False
		globalavg_layer = keras.layers.GlobalAveragePooling2D()

		inputs = keras.Input(shape=self.IMG_SHAPE)
		#x: _Any = preprocess_layer(inputs)
		x = weights(inputs, training=False)
		latent_dim = globalavg_layer(x)
		self.embedding_model: Functional = Functional(inputs=inputs, outputs=latent_dim, trainable=False)
