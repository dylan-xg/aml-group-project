"""This package handles video input and video display.

Example
-------
>>> from VideoProcessor import DisplayConfig, VideoFeed
>>> WEBCAM_SOURCE = 0
>>> def detect_face(frame): ...
>>> VideoProcessor.process_video(
... 	capture_location=WEBCAM_SOURCE,
... 	callback=detect_face,
... 	display_config=DisplayConfig.OpenCV(frametime=1/25)
... )
"""

from .DisplayConfig import (
	Headless as Headless,
	OpenCV as OpenCV,
	Jupyter as Jupyter,
	DisplayConfigType as DisplayConfigType
)

from .VideoFeed import process_video as process_video

__all__: list[str] = [
	'Headless',
	'OpenCV',
	'Jupyter',
	'DisplayConfigType',
	'process_video'
]

#__all__: list[str] = list(
#	set(DisplayConfig.__all__) |
#	set(VideoFeed.__all__)
#)
