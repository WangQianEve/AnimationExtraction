'''
Created on Jun 5, 2018
@author: qian
'''
import cv2
import numpy as np
from collections import Counter

def largestContours(area_thresh, contours):
	ret = []
	largest_area = 0
	largest = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > largest_area:
			largest = contour
			largest_area = area
		if area > area_thresh:
			ret.append(contour)
	return ret, largest

# ret: contours
def largeContours(area_thresh, contours):
	print '[largeContours] getting large contours with area threshold: %d' % area_thresh
	ret = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > area_thresh:
			ret.append(contour)
	return ret

def blur(frame, method = 'gaussian'):
	GAUSE = 5
	if method == 'gaussian':
		return cv2.GaussianBlur(frame, (GAUSE,GAUSE), 0)
	elif method == 'median':
		return cv2.medianBlur(frame, GAUSE)
	elif method == 'average':
		return cv2.blur(frame, (GAUSE,GAUSE))
	elif method == 'bilateral':
		return cv2.bilateralFilter(frame, 9, 75, 75)

def initializeWindows(names):
	for name in names:
		cv2.namedWindow(name, cv2.WINDOW_NORMAL)	
		cv2.resizeWindow(name, 480, 270)

def loadVideo(path):
	vs = cv2.VideoCapture(path)
	fps = vs.get(cv2.CAP_PROP_FPS)
	if fps < 50:
		raise Exception('fps too low', fps)
	else:
		print "[loadVideo] Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
	width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
	if height < 50:
		raise Exception('Image shape error', (width, height))
	else:
		print "[loadVideo] Video shape: width %d, height %d" % (width, height)
	return vs, fps, width, height

# ret: ratio
def getInfo(frame):
	flat = frame.flatten()
	c = Counter(flat)
	ratio = c[0] * 100 / float(np.sum(c.values()))
	print '[getInfo] blank ratio %f %%' % ratio
	return ratio

# ret: bg, thresh
def preprocessing(vs, count):
	print '[preprocessing] Start preprocessing...'
	KEY = 0
	bg = None
	thresh = 6
	n1 = 20
	n2 = 40
	ratios = []
	for frame_no in range(count):
		if not vs.isOpened():
			raise Exception('vs not opened')
		(grabbed, frame) = vs.read()
		if not grabbed:
			raise Exception('image not grabbed')
		if frame_no <= n1:
			if frame_no == 0:
				bg = frame.astype('int')
			elif frame_no == n1:
				bg = bg / frame_no
				bg = bg.astype('uint8')
				cv2.imshow("color", bg)
				bg = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
				cv2.imshow("bin", bg)
				bg = blur(bg)
				cv2.imshow("gray", bg)
				print '[preprocessing] sampling finished!'
				key = cv2.waitKey(KEY) & 0xFF
				if key == 27:
					raise Exception('Exit! i1')
				elif key == 32:
					KEY = 1 if KEY == 0 else 0
			else:
				bg += frame.astype('int')
		elif frame_no <= n2:
			gray = blur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
			sub = abs(bg.astype('int') - gray.astype('int')).astype('uint8')
			ret, binary = cv2.threshold(sub, thresh, 255, cv2.THRESH_BINARY)
			ratios.append(getInfo(binary))
			if frame_no == n2:
				ave = np.mean(ratios)
				print '[preprocessing] ratios collected! average: %f %%' % ave
				if ave < 99.9:
					raise Exception('[preprocessing] please increase threshold')
			cv2.imshow("gray", binary)
			key = cv2.waitKey(KEY) & 0xFF
			if key == 27:
				raise Exception('Exit! i2')
			elif key == 32:
				KEY = 1 if KEY == 0 else 0
	print '[preprocessing] exit successfully'
	return bg, thresh

# ret: thresh, box(x,y,w,h), color
def getMorph(binary, ksize, draw = True):
	color = None
	VISIBLE_THRESH = 1000
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
	thresh = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
	img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 2nd par: .RETR_EXTERNAL # .RETR_TREE
	large_contours, largest = largestContours(VISIBLE_THRESH, contours)
	# large_contours = largeContours(VISIBLE_THRESH, contours)
	x,y,w,h = cv2.boundingRect(largest)
	# x,y,w,h = cv2.boundingRect(np.vstack(large_contours))
	print '[getMorph] bounding box from: %d, %d, size: %d, %d' % (x,y,w,h)
	cv2.rectangle(thresh, (x,y), (x+w,y+h), 255, 2)
	if draw:
		global height, width
		color = np.zeros((height, width, 3), np.uint8)
		cv2.drawContours(color, contours, -1, (0,255,0), 3)
		cv2.drawContours(color, large_contours, -1, (0,0,255), 3)
		cv2.rectangle(color, (x,y), (x+w,y+h), (255,0,0), 2)
	return thresh, (x,y,w,h), color

def within((px, py), (x,y,w,h)):
	if px < x or px > (x+w):
		return False
	if py < y or py > (y+h):
		return False
	return True

def contoursWithin(contours, box):
	ret = []
	for contour in contours:
		if within(np.mean(contour, axis=0)[0], box):
			ret.append(contour)
	return ret

# ret: box
def getBox(binary, box):
	VISIBLE_THRESH = 50
	img, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 2nd par: .RETR_EXTERNAL # .RETR_TREE
	large_contours = largeContours(VISIBLE_THRESH, contours)
	large_contours = contoursWithin(large_contours, box)
	(x,y,w,h) = cv2.boundingRect(np.vstack(large_contours))
	cv2.rectangle(binary, (x,y), (x+w,y+h), 255, 2)
	cv2.imshow("gray", binary)
	return (x,y,w,h)

def main():
	global width, height
	readpath = './videos/'
	writepath = './output/'
	# filenames = ['blue.avi', 'white.avi', 'black.avi', 'large.avi']
	filenames = ['blue.avi', 'white.avi', 'black.avi', 'large.avi']
	filename = filenames[0]
	vs, fps, width, height = loadVideo(readpath + filename)
	initializeWindows(['gray', 'bin', 'morph', 'color_morph', 'color','sub'])
	bg, thresh = preprocessing(vs, int(fps))
	KEY = 0
	OUTPUT = False
	if OUTPUT:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(writepath + filename, fourcc, fps, (width,height))
	# parameters
	# Step 0 thresh
	# GRAY_THRESH = thresh
	# VISIBLE_THRESH = 80
	# SIZE_THRESH = 200

	# prev_frame = None
	# first = True

	frame_no = 0
	while vs.isOpened():
		(grabbed, frame) = vs.read()
		if not grabbed:
			raise Exception('[main] image not grabbed')
		frame_no += 1
		print 'frame: %d' % frame_no
		gray = blur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
		sub = abs(bg.astype('int') - gray.astype('int')).astype('uint8')
		ret, binary = cv2.threshold(sub, thresh, 255, cv2.THRESH_BINARY)
		ratio = getInfo(binary)
		if ratio > 90:
			# cv2.imshow("color", frame)
			# key = cv2.waitKey(KEY) & 0xFF
			# if key == 27:
			# 	break
			# elif key == 32:
			# 	KEY = 1 if KEY == 0 else 0
			continue
		morph, constrain_box, color_morph = getMorph(binary.copy(), ksize = 20, draw = True)
		(x,y,w,h) = getBox(binary.copy(),constrain_box)

		# cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
		cv2.imshow("color", frame)
		cv2.imshow("sub", sub)
		cv2.imshow("bin", binary)
		cv2.imshow("morph", morph)
		cv2.imshow("color_morph", color_morph)

		key = cv2.waitKey(KEY) & 0xFF
		if key == 27:
			break
		elif key == 32:
			KEY = 1 if KEY == 0 else 0

		if OUTPUT:
			out.write(frame)
	if OUTPUT:
		out.release()
	vs.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()