"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# COCO 80-class labels used by EfficientDet Lite (0-indexed class ids).
# COCO's 90 class-ID slots. Keep the 'n/a' placeholders so
# model output IDs map to the correct names.
LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'n/a', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'n/a', 'backpack', 'umbrella',
    'n/a', 'n/a', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'n/a', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'n/a', 'dining table', 'n/a',
    'n/a', 'toilet', 'n/a', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'n/a', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, threshold: float) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
    threshold: Minimum detection score to draw.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10
  box_color = (0, 0, 255)  # red

  # Initialize the object detection model
  if enable_edgetpu:
    print('WARNING: EdgeTPU is not configured for LiteRT in this script; '
          'running on CPU.')
  interpreter = Interpreter(model_path=model, num_threads=num_threads)
  interpreter.allocate_tensors()

  input_info = interpreter.get_input_details()[0]
  boxes_out, classes_out, scores_out, count_out = interpreter.get_output_details()
  input_height = input_info['shape'][1]
  input_width = input_info['shape'][2]

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, (input_width, input_height))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

    # Run object detection estimation using the model.
    interpreter.set_tensor(input_info['index'], input_tensor)
    interpreter.invoke()

    boxes = interpreter.get_tensor(boxes_out['index'])[0]
    classes = interpreter.get_tensor(classes_out['index'])[0]
    scores = interpreter.get_tensor(scores_out['index'])[0]
    count = int(interpreter.get_tensor(count_out['index'])[0])

    frame_height, frame_width = image.shape[:2]
    for i in range(min(count, len(scores))):
      score = float(scores[i])
      if score < threshold:
        continue

      ymin, xmin, ymax, xmax = boxes[i]
      x1 = max(0, min(frame_width - 1, int(xmin * frame_width)))
      y1 = max(0, min(frame_height - 1, int(ymin * frame_height)))
      x2 = max(0, min(frame_width - 1, int(xmax * frame_width)))
      y2 = max(0, min(frame_height - 1, int(ymax * frame_height)))

      class_id = int(classes[i])
      if 0 <= class_id < len(LABELS):
        label = LABELS[class_id]
      else:
        label = f'class {class_id}'
      result_text = f'{label} ({round(score, 2)})'

      cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)
      cv2.putText(image, result_text, (x1 + 10, y1 + 20),
                  cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(
      '--threshold',
      help='Minimum detection score to display.',
      required=False,
      type=float,
      default=0.3)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU), float(args.threshold))


if __name__ == '__main__':
  main()
