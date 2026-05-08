# segmenter.py
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2

def load_sam(checkpoint: str = 'sam_vit_h_4b8939.pth') -> SamPredictor:
  sam = sam_model_registry['vit_h'](checkpoint=checkpoint)
  sam.to('cuda' if __import__('torch').cuda.is_available() else 'cpu')
  return SamPredictor(sam)

def click_to_mask(
    predictor: SamPredictor,
    image: np.ndarray,          # (H, W, 3) RGB
    click_xy: tuple[int, int],  # (x, y) pixel the user clicked
) -> np.ndarray:                # (H, W) uint8 mask
  predictor.set_image(image)
  masks, scores, _ = predictor.predict(
    point_coords=np.array([click_xy]),
    point_labels=np.array([1]),         # 1 = foreground
    multimask_output=True,
  )
  best = masks[np.argmax(scores)]         # take highest-confidence mask
  return (best.astype(np.uint8)) * 255