"""
EmotionDetection: a multilingual, explainable emotion & sentiment analyzer.

    >>> from EmotionDetection import emotion_detector
    >>> emotion_detector("I just got promoted!")["primary_emotion"]
    'joy'
"""

from .emotion_detection import (EMOTIONS, LABELS, MODEL_NAME, MODEL_VERSION,
                                emotion_detector, format_legacy)

__all__ = ["emotion_detector", "format_legacy", "EMOTIONS", "LABELS",
           "MODEL_NAME", "MODEL_VERSION"]
__version__ = MODEL_VERSION
