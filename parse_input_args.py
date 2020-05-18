import argparse

def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--camera', default=0, dest='camera_num', type=int
                        , help='The system camera number, 0 is the first, 1 the second, ... ')
    parser.add_argument('--model_dir', default='ssd_mobilenet_v2_coco', dest='model_dir'
                        , help='The directory that contains the model, assumed to be in /models/[model_dir] ')
    parser.add_argument('--scale_factor', default=1, dest='scale_factor', type=float
                        , help='Resolution scale factor, default = 1')
    parser.add_argument('--confidence', default=.5, dest='confidence_threshold', type=float
                        , help='Confidence threshold to display detections , default = .5')
    parser.add_argument('--list', dest='detection_list', nargs='+'
                        , help='list of objects to detect separated by spaces')

    args = parser.parse_args()
    arg_dict = vars(args)
    return arg_dict

if __name__ == '__main__':
    d = read_args()
    print(d['scale_factor'])
    print(d['model_dir'])
    print(d['camera_num'])
    print(d['confidence_threshold'])
    print(d['detection_list'])
