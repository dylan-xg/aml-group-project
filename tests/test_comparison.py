
from pathlib import Path

import numpy as np

from panopticon.recognition._comparison import compare_faces_from_path
from src.panopticon.recognition import ExampleModel


DS_PATH = Path('data/datasets/project_face_dataset')

# verification_data/00007133.jpg verification_data/00060449.jpg 1
# verification_data/00041961.jpg verification_data/00044353.jpg 0

PHOTO1 = DS_PATH / 'verification_data/00007133.jpg'
PHOTO2 = DS_PATH / 'verification_data/00060449.jpg'
PHOTO3 = DS_PATH / 'verification_data/00041961.jpg'
PHOTO4 = DS_PATH / 'verification_data/00044353.jpg'


def test_compare_faces_euclidean() -> None:
	faces_list: list[Path] = [PHOTO1, PHOTO2, PHOTO3, PHOTO4]
	faces_array = np.array(faces_list)
	model = ExampleModel()

	result = compare_faces_from_path(
		faces=faces_array,
		image_size=model.IMG_SIZE,
		model=model,
		metric='euclidean'
	)

	expected = np.array([
		[00.00, 12.01, 12.36, 12.18],
		[12.01, 00.00, 09.42, 10.31],
		[12.36, 09.42, 00.00, 08.51],
		[12.18, 10.31, 08.51, 00.00]
	])

	np.testing.assert_allclose(actual=result, desired=expected, atol=1e-2, strict=True)


def test_compare_faces_cosine() -> None:
	faces_list: list[Path] = [PHOTO1, PHOTO2, PHOTO3, PHOTO4]
	faces_array = np.array(faces_list)
	model = ExampleModel()

	result = compare_faces_from_path(
		faces=faces_array,
		image_size=model.IMG_SIZE,
		model=model,
		metric='cosine'
	)

	expected = np.array([
		[0.00e+00, 6.83e-01, 7.87e-01, 6.93e-01],
		[6.83e-01, 0.00e+00, 5.33e-01, 5.69e-01],
		[7.87e-01, 5.33e-01, 0.00e+00, 4.24e-01],
		[6.93e-01, 5.69e-01, 4.24e-01, 0.00e+00]
	])

	np.testing.assert_allclose(actual=result, desired=expected, atol=1e-3, strict=True)
