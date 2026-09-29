import os
import cv2
import numpy as np
import logging
from app.core.config import settings

# Prefer the new MediaPipe Tasks API if available.
# Older MediaPipe releases may still expose mp.solutions.face_mesh.
try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from mediapipe.tasks.python.vision import FaceLandmarker
except (ImportError, ModuleNotFoundError):
    FaceLandmarker = None

# Configure logger
logger = logging.getLogger(__name__)

# Liveness Configuration
ML_LIVENESS_CHECK = settings.ML_LIVENESS_CHECK
# Adjusted Defaults for Real-World Webcam Usage
# Lowered to catch only severe flat colors
LIVENESS_BLUR_THRESHOLD = int(os.getenv("LIVENESS_BLUR_THRESHOLD", "10"))
# Reject high-freq noise (screen moiré)
LIVENESS_BLUR_MAX_THRESHOLD = int(os.getenv("LIVENESS_BLUR_MAX_THRESHOLD", "800"))
# Lowered for low-light scenarios
LIVENESS_COLOR_MIN_STD = float(os.getenv("LIVENESS_COLOR_MIN_STD", "5.0"))
LIVENESS_FAIL_OPEN = os.getenv("LIVENESS_FAIL_OPEN", "false").lower() == "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
_face_landmarker = None


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker
    if FaceLandmarker is None or mp is None:
        return None
    if not os.path.exists(LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            f"FaceLandmarker model not found: {LANDMARKER_MODEL_PATH}. "
            "Run 'python download_models.py' in the ml-service directory."
        )
    _face_landmarker = FaceLandmarker.create_from_model_path(LANDMARKER_MODEL_PATH)
    return _face_landmarker


def is_live(face_crop: np.ndarray) -> bool:
    """
    Check if the provided face crop represents a live person.

    This function performs multiple checks to determine liveness:
    1. **Face Mesh Validation**: Uses MediaPipe Face Mesh to ensure a valid 3D face
       structure exists.
    2. **Laplacian Variance**: Analyzes image sharpness to detect:
       - Blurriness/Flatness (Variance too low -> likely a photo/screen)
       - Excessive Noise/Moiré patterns (Variance too high -> likely a screen capture)
    3. **Color Standard Deviation**: Checks for sufficient color diversity to filter
       out low-quality spoofs or flat masks.

    Args:
        face_crop (np.ndarray): The cropped face image in BGR format (OpenCV default).

    Returns:
        bool: True if the face passes checks or checks are disabled/fail-open.
              False if a potential spoof is detected.
    """
    if not ML_LIVENESS_CHECK:
        return True

    if face_crop is None or face_crop.size == 0:
        return False

    # Ensure image is in RGB for MediaPipe
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # --- Quality Checks ---
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Range check on Variance:
    # - Too low (<10): Likely solid color, extremely blurry, or flat mask.
    # - Too high (>800): Likely screen moiré, printed halftone, or excessive noise.
    if variance < LIVENESS_BLUR_THRESHOLD:
        logger.warning(
            f"Spoof detected: Variance TOO LOW. "
            f"Score={variance:.2f} < {LIVENESS_BLUR_THRESHOLD}"
        )
        return False

    if variance > LIVENESS_BLUR_MAX_THRESHOLD:
        logger.warning(
            f"Spoof detected: Variance TOO HIGH (Screen Artifacts?). "
            f"Score={variance:.2f} > {LIVENESS_BLUR_MAX_THRESHOLD}"
        )
        return False

    # Color Diversity Check
    (mean, std) = cv2.meanStdDev(face_crop)
    avg_std = np.mean(std)

    if avg_std < LIVENESS_COLOR_MIN_STD:
        logger.warning(
            f"Spoof detected: Low color diversity (Flat/Low Light). "
            f"StdDev={avg_std:.2f} < {LIVENESS_COLOR_MIN_STD}"
        )
        return False

    # Log passing values for debugging
    logger.info(
        f"Liveness Checks Passed: Variance={variance:.2f}, StdDev={avg_std:.2f}"
    )

    try:
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            ) as face_mesh:
                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    logger.warning("Spoof detected: No face mesh constructed.")
                    return False

                return True

        landmarker = _get_face_landmarker()
        if landmarker is None:
            logger.warning(
                "No compatible MediaPipe face mesh/landmarker available. "
                "Skipping mesh-based liveness check."
            )
            return True

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            logger.warning("Spoof detected: No face landmarks detected.")
            return False

        return True

    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return True if LIVENESS_FAIL_OPEN else False
