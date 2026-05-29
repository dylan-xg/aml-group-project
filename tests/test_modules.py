import logging
from pathlib import Path

from panopticon.modules import KerasModule


TEST_MODEL = Path("src/panopticon/model_weights/expression9_orig_longrun.keras")


def test_module_instance():
	module = KerasModule(path=TEST_MODEL)
	assert module.enabled == False
	assert module.name == TEST_MODEL.stem
	assert module.model == None


def test_module_load():
	module = KerasModule(path=TEST_MODEL)
	module.load_model()
	assert module.model is not None
	logging.info(module.model.summary(show_trainable=True))
