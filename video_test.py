import numpy as np
import cv2
import tensorflow as tf
import os

# local files
import parse_input_args as prs

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

def annotate_image(image, output_dict, conf_level, object_list, frame_rate):
    y = image.shape[0]
    x = image.shape[1]
    # bounding boxes represent the distance of the box border from the
    # [top, left, bottom, right] respectively
    # in cv2, point 1 is the top left and pt is the bottom right

    line_thickness = 1
    font_fam = cv2.FONT_HERSHEY_SIMPLEX
    font_size = .3
    for det in range(output_dict['num_detections']):
        object_index = output_dict['detection_classes'][det]
        object_label = labels[object_index]
        object_score = output_dict['detection_scores'][det]
        fr_text = 'frame rate: {: .1}'.format(et)
        sz = cv2.getTextSize(fr_text, font_fam, .4, 1)
        cv2.putText(image, fr_text, (int(x * .01), int(y * .99)), font_fam, .4, (0, 0, 0), 1)

        if bool(object_score >= conf_level) & bool(object_label in object_list):
            box = output_dict['detection_boxes'][det]
            x1 = int(x * box[1])
            y1 = int(y * box[0])
            x2 = int(x * box[3])
            y2 = int(y * box[2])
            cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 0), line_thickness)

            text = '{}: {: .1%}'.format(object_label, object_score)
            sz = cv2.getTextSize(text, font_fam, font_size, line_thickness)
            baseline = sz[1]
            # box around text
            cv2.rectangle(image, (x1, y1), (x1 + sz[0][0], y1 - sz[0][1] - line_thickness), (200, 200, 0), -1)
            cv2.putText(image, text, (x1, y1 - line_thickness), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1)

    return image

def get_labels(path):
    with open(labels_path, 'rb') as f:
        labels_list = f.readlines()

    idx = 0
    labels_dict = {}
    for cat in labels_list:
        l = cat.rstrip()
        labels_dict[idx] = l.decode('utf-8')
        idx = idx + 1

    return labels_dict

args = prs.read_args()

# open a TF model
model_dir = args['model_dir']
model_path = os.path.join(os.getcwd(), 'models/{}/saved_model'.format(model_dir))
tf_model = tf.saved_model.load(model_path)
tf_model = tf_model.signatures['serving_default']

labels_path = os.path.join(os.getcwd(), 'models/labelmap.txt')
labels = get_labels(labels_path)

detection_list = args['detection_list']
if detection_list is None:
    detection_list = labels
conf_th = args['confidence_threshold']

cap = cv2.VideoCapture(args['camera_num'])
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    start = cv2.getTickCount()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # resize the window
    dim = (int(frame.shape[1] * args['scale_factor'])
           , int(frame.shape[0] * args['scale_factor']))
    frame = cv2.resize(frame, dim)

    # Our operations on the frame come here
    img = np.array(frame)
    od = run_inference_for_single_image(tf_model, img)
    end = cv2.getTickCount()
    et = (end - start) / cv2.getTickFrequency()

    annotated_frame = annotate_image(frame, od, conf_level=conf_th, object_list=detection_list, frame_rate=et)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
# release the capture
cap.release()
cv2.destroyAllWindows()