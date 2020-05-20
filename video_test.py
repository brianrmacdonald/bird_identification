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
    new_image = image
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
        new_image = cv2.putText(new_image, fr_text, (int(x * .01), int(y * .99)), font_fam, .4, (0, 0, 0), 1)

        if bool(object_score >= conf_level) & bool(object_label in object_list):
            box = output_dict['detection_boxes'][det]
            x1 = int(x * box[1])
            y1 = int(y * box[0])
            x2 = int(x * box[3])
            y2 = int(y * box[2])
            new_image = cv2.rectangle(new_image, (x1, y1), (x2, y2), (200, 200, 0), line_thickness)

            text = '{}: {: .1%}'.format(object_label, object_score)
            sz = cv2.getTextSize(text, font_fam, font_size, line_thickness)
            baseline = sz[1]
            # box around text
            new_image = cv2.rectangle(new_image, (x1, y1), (x1 + sz[0][0], y1 - sz[0][1] - line_thickness), (200, 200, 0), -1)
            new_image = cv2.putText(new_image, text, (x1, y1 - line_thickness), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1)

    return new_image

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

def zoom(image, level):
    y_dim = image.shape[0]
    x_dim = image.shape[1]
    zoom_inc_x = x_dim * .1
    zoom_inc_y = y_dim * .1
    crop_x = zoom_inc_x * level
    crop_y = zoom_inc_y * level
    start_x = int(crop_x/2)
    start_y = int(crop_y/2)
    end_x = int(x_dim - crop_x)
    end_y = int(y_dim - crop_y)
    cropped_image = image[start_y:end_y, start_x:end_x]
    final_image = cv2.resize(cropped_image, (x_dim, y_dim), interpolation=cv2.INTER_CUBIC)

    return final_image

def magnify(image, mag_level):
    y_dim = image.shape[0]
    x_dim = image.shape[1]
    mag_inc_x = x_dim * .1
    mag_inc_y = y_dim * .1
    mag_x = int(mag_inc_x * mag_level + x_dim)
    mag_y = int(mag_inc_y * mag_level + y_dim)
    final_image = cv2.resize(image, (mag_x, mag_y), interpolation=cv2.INTER_CUBIC)

    return final_image

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
    detection_list = list(labels.values())
conf_th = args['confidence_threshold']

scale_level = 0
zoom_level = 0
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
    new_img = frame
    if zoom_level > 0:
        new_img = zoom(frame, zoom_level)

    if scale_level != 0:
        new_img = magnify(new_img, scale_level)

    # Our operations on the frame come here
    img = np.array(new_img)
    od = run_inference_for_single_image(tf_model, img)
    end = cv2.getTickCount()
    et = (end - start) / cv2.getTickFrequency()

    annotated_frame = annotate_image(new_img, od, conf_level=conf_th, object_list=detection_list, frame_rate=et)

    # Display the resulting frame
    cv2.imshow('Camera {}'.format(args['camera_num']), annotated_frame)

    # frame operations
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('z'):
        zoom_level += 1
    elif k == 26:  # ctrl-z
        zoom_level -= 1
        if zoom_level < 0:
            zoom_level = 0
    elif k == 109:  # lowercase m
        scale_level += 1
    elif k == 13:  # ctrl m
        scale_level -= 1
# release the capture
cap.release()
cv2.destroyAllWindows()