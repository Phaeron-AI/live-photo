"""
quick_select.py — Click to select object mask using SAM.

Controls:
  Left click     — add a region to the mask
  Right click    — remove a region from the mask
  S              — save and exit
  C              — clear all and start over
  Q              — quit without saving

For multi-colour objects (flags, patterned cloth):
  Click each colour stripe separately — they merge into one mask.
"""

import cv2
import numpy as np
import sys
from segmenter import load_sam, click_to_mask


def select_and_save(image_path: str, mask_path: str = 'mask.png'):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: could not read {image_path}")
        return

    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor  = load_sam()

    # Accumulated mask — OR of all clicked regions
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    window = 'Left click = add | Right click = remove | S = save | C = clear | Q = quit'

    def redraw():
        preview = image_bgr.copy()
        if combined_mask.any():
            overlay = preview.copy()
            overlay[combined_mask > 0] = (0, 200, 100)
            cv2.addWeighted(overlay, 0.4, preview, 0.6, 0, preview)
            # Draw white border around mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview, contours, -1, (255, 255, 255), 1)
        white_px = (combined_mask > 0).sum()
        pct      = 100 * white_px / combined_mask.size
        cv2.putText(preview, f"Selected: {white_px:,} px ({pct:.1f}%)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(preview, f"Selected: {white_px:,} px ({pct:.1f}%)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow(window, preview)

    def on_click(event, x, y, flags, param):
        nonlocal combined_mask

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Adding region at ({x}, {y})...")
            mask = click_to_mask(predictor, image_rgb, (x, y))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            print(f"  Total selected: {(combined_mask > 0).sum():,} pixels")
            redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"Removing region at ({x}, {y})...")
            mask = click_to_mask(predictor, image_rgb, (x, y))
            # Remove this region from the combined mask
            combined_mask = cv2.bitwise_and(combined_mask,
                                            cv2.bitwise_not(mask))
            print(f"  Total selected: {(combined_mask > 0).sum():,} pixels")
            redraw()

    cv2.imshow(window, image_bgr)
    cv2.setMouseCallback(window, on_click)
    print("Left click each part of the object to build up the mask.")
    print("For a tri-colour flag: click each stripe separately.")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            if not combined_mask.any():
                print("No mask yet — click the object first")
                continue
            cv2.imwrite(mask_path, combined_mask)
            print(f"\nSaved → {mask_path}")
            check = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            print(f"Verified: {(check > 0).sum():,} white pixels")
            break

        elif key == ord('c'):
            combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            print("Cleared — start over")
            redraw()

        elif key == ord('q'):
            print("Quit without saving")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    select_and_save(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else 'mask.png'
    )