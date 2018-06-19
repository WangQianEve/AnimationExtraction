'''
Created on Jun 4, 2018
@author: qian
'''
import cv2

def clipVideo():
	DISPLAY = True
	OBSERVE = False
	OUTPUT = True if not OBSERVE else False
	recording = False
	readpath = './videos/'
	writepath = './newVideos/'
	filename = 'blue.avi'
	vs = cv2.VideoCapture(readpath + filename)
	fps = vs.get(cv2.CAP_PROP_FPS)
	print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
	width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print "Video shape: width %d, height %d" % (width, height)
	if OUTPUT:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(writepath + filename, fourcc, fps, (width,height))
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)	
	cv2.resizeWindow('image', 800, 450)
	frame_no = 0
	while vs.isOpened():
		(grabbed, frame) = vs.read()
		if not grabbed:
			break
		if OBSERVE: print frame_no
		if DISPLAY:
			cv2.imshow("image", frame)
			time = 0 if OBSERVE else 1
			key = cv2.waitKey(time) & 0xFF
			if key == 27:
				break
			if OUTPUT:
				if frame_no == 20:
					recording=True
					print ("recording")
				if frame_no == 430:
					recording = False
					print ("end")
					break
		if OUTPUT and recording:
			out.write(frame)
		frame_no += 1
	if OUTPUT:
		out.release()
	vs.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	clipVideo()