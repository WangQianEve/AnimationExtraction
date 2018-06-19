'''
Created on Jun 4, 2018
@author: qian
'''
import cv2
import numpy as np

def largeContours(area_thresh,contours):
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
    return ret, largest, largest_area

# def closeContours(contours):

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

def main():
	DISPLAY = True
	OUTPUT = False
	recording = False

	readpath = './newVideos/'
	writepath = './subVideos/'
	filenames = ['blue.avi', 'white.avi', 'black.avi', 'large.avi']
	filename = filenames[3]
	vs = cv2.VideoCapture(readpath + filename)

	fps = vs.get(cv2.CAP_PROP_FPS)
	print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
	width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print "Video shape: width %d, height %d" % (width, height)

	if OUTPUT:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(writepath + filename, fourcc, fps, (width,height))

	# windows = ['image', 'sub', 'close', 'open', 'blank']

	# ratio = 0.25
	color_ratio = 0.6
	gray_ratio = 0.3
	# width = int(1600 * ratio)
	# height = int(900 * ratio)

	cv2.namedWindow('color', cv2.WINDOW_NORMAL)	
	cv2.resizeWindow('color', (int(width * color_ratio), int(height * color_ratio)))
	cv2.namedWindow('gray', cv2.WINDOW_NORMAL)	
	cv2.resizeWindow('gray', (int(width * gray_ratio), int(height * gray_ratio)))

	# parameters
	# Step 0 thresh
	GRAY_THRESH = 5
	VISIBLE_THRESH = 80
	SIZE_THRESH = 200

	frame_no = 0
	prev_frame = None
	first = True
	while vs.isOpened():
		(grabbed, color) = vs.read()
		if not grabbed:
			break

		frame = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
		frame = blur(frame)

		if first:
			prev_frame = frame
			first = False
			continue
		print frame_no

		# Step 0 : thresh sub
		sub = abs(prev_frame.astype('int') - frame.astype('int')).astype('uint8')
		ret, thresh = cv2.threshold(sub, GRAY_THRESH, 255, cv2.THRESH_BINARY) # should small so that black or white backgroudn is ok

		blank = np.zeros((height, width, 3), np.uint8)

		# Step 1 : visible contours
		img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # .RETR_EXTERNAL # .RETR_TREE
		large_contours, largest_contour, largest_area = largeContours(VISIBLE_THRESH, contours)
		print 'color largest area: %f' % largest_area
		if largest_area >= VISIBLE_THRESH:
			x,y,w,h = cv2.boundingRect(largest_contour)
			cv2.rectangle(color, (x,y), (x+w,y+h), (0,255,0), 2) # green
		cv2.drawContours(color, large_contours, -1, (0,0,255), 3)
		cv2.drawContours(blank, large_contours, -1, (0,0,255), 3)

		# Step 2 : connect
		kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
		kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
		main_thresh1 = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel1)
		main_thresh2 = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel2)

		# Step : choose large
		# Step : bouding box
		# print blank.shape
		# print main_thresh1.shape
		# print main_thresh2.shape

		if DISPLAY:
			color_imgs = [color, blank, main_thresh1, main_thresh2]
			for canvas in color_imgs[1:]:
				gray_canvas = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
				img, contours, hierarchy = cv2.findContours(gray_canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # .RETR_EXTERNAL # .RETR_TREE
				large_contours, largest_contour, largest_area = largeContours(SIZE_THRESH, contours)
				print 'largest area: %f' % largest_area
				if largest_area >= SIZE_THRESH:
					x,y,w,h = cv2.boundingRect(np.vstack(large_contours))
					cv2.rectangle(canvas, (x,y), (x+w,y+h), (0,255,0), 2) # green

			row1 = np.hstack(color_imgs[:2])
			row2 = np.hstack(color_imgs[2:])
			color_img = np.vstack((row1, row2))
			cv2.imshow("color", color_img)
			cv2.imshow("gray", thresh)
			key = cv2.waitKey(0) & 0xFF
			if key == 27:
				break
		if OUTPUT:
			out.write(frame)
		frame_no += 1
		prev_frame = frame
		# prev_frames.append(frame)
		# prev_frames = prev_frames[1:]
	if OUTPUT:
		out.release()
	vs.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()