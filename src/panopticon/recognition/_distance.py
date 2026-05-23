"""Distance functions."""

import numpy as np
import numpy.typing as npt


def euclidean_distance(
	x: npt.NDArray[np.float32],
	y: npt.NDArray[np.float32]
) -> float:
	dist = np.linalg.norm(x-y)
	return float(dist)
