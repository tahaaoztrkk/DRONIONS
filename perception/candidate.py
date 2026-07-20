"""
=========================================================
Detection Candidate
=========================================================

This module defines the data structure used by the
perception pipeline.

Every detected object is represented as a DetectionCandidate.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DetectionCandidate:
    """
    Represents one detected object.
    """

    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    label: str

    confidence: float

    bbox: Tuple[int, int, int, int]

    prompt: str

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    pixel_area: float = 0.0

    relative_area: float = 0.0

    aspect_ratio: float = 1.0

    center: Tuple[int, int] = (0, 0)

    distance_to_center: float = 0.0

    # --------------------------------------------------
    # Scores
    # --------------------------------------------------

    geometry_score: float = 0.0

    semantic_score: float = 0.0

    center_score: float = 0.0

    final_score: float = 0.0

    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------

    track_id: Optional[int] = None

    age: int = 0

    frames_seen: int = 1

    # --------------------------------------------------
    # Verification
    # --------------------------------------------------

    verified: bool = False

    gemini_verified: bool = False

    description: str = ""

    # --------------------------------------------------
# Image Information
# --------------------------------------------------

    image_width: int = 0

    image_height: int = 0

    width: int = 0

    height: int = 0

    normalized_center: Tuple[float, float] = (0.0, 0.0)

    normalized_bbox: Tuple[float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    0.0,)

    timestamp: float = 0.0

    # --------------------------------------------------

    metadata: dict = field(default_factory=dict)