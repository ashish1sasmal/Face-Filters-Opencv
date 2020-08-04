from blend import alpha_blend
import numpy as np

def overlay_image(bg, fg, fgMask, coords):
	(sH, sW) = fg.shape[:2]
	(x, y) = coords
	overlay = np.zeros(bg.shape, dtype="uint8")
	overlay[y:y + sH, x:x + sW] = fg

	alpha = np.zeros(bg.shape[:2], dtype="uint8")
	alpha[y:y + sH, x:x + sW] = fgMask
	alpha = np.dstack([alpha] * 3)

	output = alpha_blend(overlay, bg, alpha)

	return output
