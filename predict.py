import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Union
import math
import time
import matplotlib.pyplot as plt
from heart_rate import *
import os
import threading
import sys
import signal
import matplotlib.patches as mpatch

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
bbox_increase_x = 10
bbox_increase_y = 25
prev_start_point = 0
prev_end_point = 0
first_time_detect = True
Classifier_used = 'None'


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')


def _normalized_to_pixel_coordinates(
		normalized_x: float, normalized_y: float, image_width: int,
		image_height: int) -> Union[None, Tuple[int, int]]:
	"""Converts normalized value pair to pixel coordinates."""

	# Checks if the float value is between 0 and 1.
	def is_valid_normalized_value(value: float) -> bool:
		return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

	if not (is_valid_normalized_value(normalized_x) and
					is_valid_normalized_value(normalized_y)):
		# TODO: Draw coordinates even if it's outside of the image bounds.
		return None
	x_px = min(math.floor(normalized_x * image_width), image_width - 1)
	y_px = min(math.floor(normalized_y * image_height), image_height - 1)
	return x_px, y_px

def visualize(
		image,
		detection_result
) -> np.ndarray:
	"""Draws bounding boxes and keypoints on the input image and return it.
	Args:
		image: The input RGB image.
		detection_result: The list of all "Detection" entities to be visualize.
	Returns:
		Image with bounding boxes.
	"""
	global prev_start_point
	global prev_end_point
	global first_time_detect
	global Classifier_used
	#global prev_annotated_image
	annotated_image = image.copy()
	height, width, _ = image.shape
	start_point = -1, -1
	end_point = -1, -1

	image_gray = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY)

	num_detections = 0

	for detection in detection_result.detections:
		#print('inside num_detections')
		# Draw bounding_box
		bbox = detection.bounding_box
		start_point = max(0, bbox.origin_x - bbox_increase_x), max(0,bbox.origin_y - bbox_increase_y)
		end_point = min(annotated_image.shape[1] - 1, bbox.origin_x + bbox.width + bbox_increase_x), min(annotated_image.shape[0] - 1, bbox.origin_y + bbox.height + bbox_increase_y)
		cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

		# Draw keypoints
		for keypoint in detection.keypoints:
			keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
			color, thickness, radius = (0, 255, 0), 2, 2
			cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

		# Draw label and score
		category = detection.categories[0]
		category_name = category.category_name
		category_name = '' if category_name is None else category_name
		probability = round(category.score, 2)
		result_text = category_name + ' (' + str(probability) + ')'
		text_location = (MARGIN + bbox.origin_x,
										 MARGIN + ROW_SIZE + bbox.origin_y)
		cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
								FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

		num_detections += 1
		Classifier_used = 'MediaPipe Blaze'
		first_time_detect = False
	
	#if cascade_detection == 0 and first_time_detect == False and num_detections == 0:
	# use of previous values if no face detected
	if first_time_detect == False and num_detections == 0:
		start_point = prev_start_point
		end_point = prev_end_point
		cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

	prev_start_point = start_point
	prev_end_point = end_point

	return annotated_image, start_point, end_point, num_detections, Classifier_used

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

model_path = 'blaze_face_short_range.tflite'

face_detection_result = []
frames_processed = 0
rgb_annotated_image = None
rgb_image_face = None

state = load_state('state.json')
model = load_model('model.onnx')
timestamps = []
bvp = []
mabvplist = []
hr = 0
avghr = 0
snr = 0
hrlist = []
mahrlist = []
first_time = 1
timestamp_start = 0
windowsize = 6
windowsize_bvp = 1
nvals_plot = 64
n_hr_bvp = 64
lock = threading.Lock()
stop_event = threading.Event()
filtered_data = 0
avgsigqual = 0
signal_quality  = 0



# callback function for mediapipe face detector
def print_result(result, output_image, timestamp_ms):
		#print('face detector result: {}'.format(result))
		if not stop_event.is_set():
			global face_detection_result
			global rgb_annotated_image
			global rgb_image_face
			global bbox_increase
			global model 
			global state
			global timestamp_start
			global first_time
			global filtered_data
			global avgsigqual
			global signal_quality
			face_detection_result = result
			image_copy = np.copy(mp_image.numpy_view())
			annotated_image, start_point, endpoint, num_detections, classifier_used = visualize(image_copy, face_detection_result)
			rgb_annotated_image = annotated_image
			if start_point[0] != -1 and endpoint[0] != -1:
				image_face = image_copy[start_point[1]: endpoint[1],
				start_point[0]: endpoint[0]] 
				#rgb_image_face = cv2.cvtColor(image_face, cv2.COLOR_BGR2RGB)
				rgb_image_face = image_face
				rgb_image_face_copy = cv2.resize(rgb_image_face, (36, 36), interpolation=cv2.INTER_AREA)
				#print('rgb_image_face.shape: ', rgb_image_face.shape)
				rgb_image_face_copy = rgb_image_face_copy.astype(np.float32)
				output, state = model(rgb_image_face_copy, state)
				#print('output: ', output)
				with lock:
					bvp.append(output)
					if first_time == 1:
						timestamps.append(0)
						timestamp_start = time.time()
						first_time = 0
					else:
						timestamps.append(time.time() - timestamp_start)
					if len(bvp) > 1:

						sr = min(len(bvp), n_hr_bvp)/(timestamps[-1] - timestamps[-min(len(bvp), n_hr_bvp)])  # computing current sampling rate 
						#(based on the time nedded to process input frames)

						hr, snr = get_hr(bvp[-n_hr_bvp:], sr=sr)    # computing hear rate
						print(f'\nSampling rate over last {min(len(bvp), n_hr_bvp)} values: {sr:.2f} Hz')
						#print('Classifier_used: ', Classifier_used)
						if bool(num_detections): 
							print('Face detected: ', bool(num_detections))
						else:
							print('Face detected: ', bool(num_detections), ' previous detection used')
						#hr, snr, filtered_data = preprocess_get_hr(bvp[-n_hr_bvp:], sr=sr)
						print(f'Heart rate: {hr:.2f}')
						
						#print(f'snr: {snr:.2f}')

						# computing signal quality 

						if snr == 0:
							signal_quality = 0
						elif snr == -1:
							signal_quality = 1
						else:
							signal_quality = 1 - 1/(1 + snr)
						avgsigqual = (avgsigqual*(len(bvp) - 2) + signal_quality)/(len(bvp) - 1)
						hrlist.append(hr)
						print(f'Signal_quality: {signal_quality:.2f}')
						#print(f'Avg signal quality: {avgsigqual:.2f}')

						# comouting moving average for heart rate

						if len(hrlist) >= windowsize:
							ma = np.mean(np.array(hrlist[-windowsize:]))
							mahrlist.append(ma)
						else:
							ma = np.mean(np.array(hrlist))
							mahrlist.append(ma)




options = FaceDetectorOptions(
		base_options=BaseOptions(model_asset_path=model_path),
		running_mode=VisionRunningMode.LIVE_STREAM,
		result_callback=print_result)

cap = cv2.VideoCapture(0)
frame_no = 0
start_time = 0
img_num = 0

plt.ion()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), constrained_layout=True)
# Plot initial empty data and get the line artist object
line, = ax1.plot([], [])
line2, = ax2.plot([], [])
plt.title("Heart Rate and BVP", pad=25)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('BVP')
ax1.set_title('BVP vs Time (s)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Heart Rate (BPM)')
ax2.set_title('Heart Rate (BPM) vs Time (s)')
ax2.set_ylim(0,160)
#hrlist = []
#mahrlist = []


def signal_handler(sig, frame):
    avghr = np.mean(np.array(mahrlist))
    hr = mahrlist[-1]
    print(f'Heart Rate is {hr:.2f}')
    print(f'Average Heart Rate is {avghr:.2f}')
    print(f'Average Signal Quality is {avgsigqual:.2f}')
    print(f'Session Length is {timestamps[-1]:.2f} s')
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()		
    plt.close('all')
    plt.figure()
    plt.text(0.5, 0.80, f'Average Heart Rate is {avghr:.2f} bpm',
    fontsize=14, ha='center', va='center',bbox=dict(facecolor='yellow', alpha=0.5, boxstyle='round,pad=0.5'))
    plt.text(0.5, 0.20, f'Average Signal quality is {avgsigqual:.2f}',
    fontsize=14, ha='center', va='center',bbox=dict(facecolor='yellow', alpha=0.5, boxstyle='round,pad=0.5'))
    plt.text(0.5, 0.50, f'Session Length is {timestamps[-1]:.2f} secs',
    fontsize=14, ha='center', va='center',bbox=dict(facecolor='yellow', alpha=0.5, boxstyle='round,pad=0.5'))
    plt.show(block=True)
    sys.exit()

# Register the signal handler at the beginning of your script
signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":

	with FaceDetector.create_from_options(options) as detector:
		try: 
			while cap.isOpened():
				try: 
					if frame_no == 0:
						frame_no = 1
						start_time = time.time()

					success, image = cap.read()
					image.flags.writeable = False

					if success and not stop_event.is_set(): 
					
					
						mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
						image = np.copy(mp_image.numpy_view())


						frame_timestamp_ms = int((time.time() - start_time)*1000)

						#mediapipe face detection thread 
						detector.detect_async(mp_image, frame_timestamp_ms)

						if rgb_annotated_image is not None:
							cv2.imshow('MediaPipe Face Detection', rgb_annotated_image)
							if cv2.waitKey(5) & 0xFF == 27: # Press 'Esc' to exit
								break
						if rgb_image_face is not None:
							cv2.imshow('cropped face: ', rgb_image_face)
							if cv2.waitKey(5) & 0xFF == 27: # Press 'Esc' to exit
								break

	        
					#if (len(bvp) > 1) and len(timestamps) == len(bvp) and len(timestamps) == (len(mahrlist) + 1):
						
						# plotting results here
						if (len(bvp) > 1):

							if cv2.waitKey(1) & 0xFF == ord('q'):
								break
							with lock:
								avghr = np.mean(np.array(mahrlist))
								hr = mahrlist[-1]

								line.set_data(timestamps[-nvals_plot:], bvp[-nvals_plot:])
								#line.set_data(timestamps[-nvals_plot:], filtered_data[-nvals_plot:])
								line2.set_data(timestamps[1:][-nvals_plot:], mahrlist[-nvals_plot:])

								ax1.relim() # Recalculate limits
								ax1.autoscale_view()
								ax2.relim() # Recalculate limits
								ax2.set_ylim(0,200)
								ax2.autoscale_view()

							ax1.text(0.05, 1.30, f'Heart Rate = {hr:.2f}', 
		            verticalalignment='top', horizontalalignment='left',
		            transform=ax1.transAxes,
		            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5), fontsize=14)
							ax1.text(0.40, 1.30, f'Signal Quallity = {signal_quality:.2f}', 
		            verticalalignment='top', horizontalalignment='left',
		            transform=ax1.transAxes,
		            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5), fontsize=14)
							ax1.text(0.70, 1.30, f'Avg. Heart Rate = {avghr:.2f}', 
		            verticalalignment='top', horizontalalignment='left',
		            transform=ax1.transAxes,
		            bbox=dict(boxstyle='square', facecolor='white', alpha=0.5), fontsize=14)

							plt.draw()

							plt.pause(0.1)

					elif not success:
						continue
				except KeyboardInterrupt:
					print('in interrupt 1')
					avghr = np.mean(np.array(mahrlist))
					hr = mahrlist[-1]
					print(f'Heart Rate is {hr:.2f}')
					print(f'Average Heart Rate is {avghr:.2f}')
					stop_event.set()
				#finally:
					detector.close()
					cap.release()
					#cv2.destroyAllWindows()		
					sys.exit()
					break
				#sys.exit()
				
		except KeyboardInterrupt:
			print('in interrupt 2')
			avghr = np.mean(np.array(mahrlist))
			hr = mahrlist[-1]
			print(f'Heart Rate is {hr:.2f}')
			print(f'Average Heart Rate is {avghr:.2f}')
			#sys.exit()
		finally:
			stop_event.set()
			detector.close()
			cap.release()
			#cv2.destroyAllWindows()		
			sys.exit()

	plt.ioff()
	plt.show()

	cap.release()    