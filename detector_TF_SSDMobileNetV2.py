import argparse
import cv2 as cv
from motrackers.detectors import TF_SSDMobileNetV2

parser = argparse.ArgumentParser(
    description='Object detections in input video using TensorFlow model of MobileNetSSD.')

parser.add_argument(
    '--source_path', '-v', type=str, default="Input/zero.mp4", help='Input video path.')
parser.add_argument(
    '--weights', '-w', type=str,
    default="pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    help='path to weights file of tf-MobileNetSSD (`.pb` file).')
parser.add_argument(
    '--config', '-c', type=str,
    default="pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
    help='path to config file of Caffe-MobileNetSSD (`.pbtxt` file).')
parser.add_argument(
    '--labels', '-l', type=str,
    default="pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_names.json",
    help='path to labels file of coco dataset (`.names` file.)')
parser.add_argument(
    '--gpu', type=bool, default=False,
    help='Flag to use gpu to run the deep learning model. Default is `False`')

args = parser.parse_args()

model = TF_SSDMobileNetV2 (
    weights_path=args.weights,
    configfile_path=args.config,
    labels_path=args.labels,
    confidence_threshold=0.5,
    nms_threshold=0.2,
    draw_bboxes=True,
    use_gpu=args.gpu)

# Create a VideoCapture object
cap = cv.VideoCapture(args.source_path)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the video's width, height, and frames per second (fps)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv.VideoWriter('Output/result.mp4', fourcc, fps, (frame_width, frame_height))

# Read and display frames from the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break  # Break the loop if the end of the video is reached
    
    bboxes, confidences, class_ids = model.detect(frame)
    updated_image = model.draw_bboxes(frame, bboxes, confidences, class_ids)
    
    # Write the frame to the output video file
    out.write(updated_image)  # use updated_image if the processed frame is stored in this variable

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all OpenCV windows
cv.destroyAllWindows()