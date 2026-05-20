"""This package handles video input and video display.

Example
-------
>>> from panopticon import video_processor as VP
>>> WEBCAM_SOURCE = 0
>>> def detect_face(frame): ...
>>> VP.process_video(
... 	capture_location=WEBCAM_SOURCE,
... 	callback=detect_face,
... 	display_config=VP.DisplayConfig.OpenCV(frametime=1/25)
... )
"""

#from ._display_config import (
#	Headless as Headless,
#	OpenCV as OpenCV,
#	Jupyter as Jupyter,
#	DisplayConfigType as DisplayConfigType
#)

from ._video_feed import (
	VideoFeed as VideoFeed,
	process_video as process_video
)

from . import _display_config as DisplayConfig

__all__: list[str] = [
	'VideoFeed',
	'DisplayConfig',
	'process_video'
]

#__all__: list[str] = [
#	'Headless',
#	'OpenCV',
#	'Jupyter',
#	'DisplayConfigType',
#	'process_video'
#]

#__all__: list[str] = list(
#	set(DisplayConfig.__all__) |
#	set(VideoFeed.__all__)
#)
