# import the necessary packages
from flask import Response
from flask import Flask
from flask import render_template
import threading
import datetime
import time
import cv2
import argparse
import logging

from edgetpumodel import EdgeTPUModel
from utils import get_image_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
time.sleep(2.0)


@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


def detect_stream(model: EdgeTPUModel, video_device: str):
	# grab global references to the video stream, output frame, and
	# lock variables
	global outputFrame, lock
	# initialize the motion detector and the total number of frames
	# read thus far
	input_size = model.get_image_size()
	cam = cv2.VideoCapture(video_device)

	while True:
		try:
			res, image = cam.read()

			if res is False:
				logger.error("Empty image received")
				break
			else:
				full_image, net_image, pad = get_image_tensor(image, input_size[0])
				pred = model.forward(net_image)
				frame = model.process_images(pred[0], full_image, pad)

				timestamp = datetime.datetime.now()
				tinference, tnms = model.get_last_inference_time()
				logger.info("Frame done in {}".format(tinference + tnms))
				cv2.putText(image, timestamp.strftime(
					"%A %d %B %Y %I:%M:%S%p"), (10, image.shape[0] - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

				with lock:
					outputFrame = frame.copy()

		except KeyboardInterrupt:
			break

	cam.release()


def append_objs_to_img(image, inference_size, objs, labels):
	height, width, channels = image.shape
	scale_x, scale_y = width / inference_size[0], height / inference_size[1]
	for obj in objs:
		bbox = obj.bbox.scale(scale_x, scale_y)
		x0, y0 = int(bbox.xmin), int(bbox.ymin)
		x1, y1 = int(bbox.xmax), int(bbox.ymax)

		percent = int(100 * obj.score)
		label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

		image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
		image = cv2.putText(image, label, (x0, y0+30),
							cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
	return image


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution


if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, default='0.0.0.0', help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, default='4664', help="port number of the server (1024 to 65535)")
	ap.add_argument("--model", "-m", help="weights file", required=True)
	ap.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
	ap.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
	ap.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
	ap.add_argument("--device", type=int, default=1, help="Image capture device to run live detection")
	ap.add_argument("--quiet", "-q", action='store_true', help="Disable logging (except errors)")
	ap.add_argument("--v8", action='store_true', help="yolov8 model?")

	args = ap.parse_args()
	# start a thread that will perform motion detection
	if args.quiet:
		logging.disable(logging.CRITICAL)
		logger.disabled = True

	logger.info("Opening stream on device: {}".format(args.device))
	model = EdgeTPUModel(args.model, args.names, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh, v8=args.v8)

	t = threading.Thread(target=detect_stream, args=(model, args.device))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args.ip, port=args.port, debug=True, threaded=True, use_reloader=False)
