import cv2

def alpha_blend(fg, bg, alpha):


	fg = fg.astype("float")
	bg = bg.astype("float")
	alpha = alpha.astype("float") / 255
	# perform alpha blending
	fg = cv2.multiply(alpha, fg)
	bg = cv2.multiply(1 - alpha, bg)

	output = cv2.add(fg, bg)

	return output.astype("uint8")
