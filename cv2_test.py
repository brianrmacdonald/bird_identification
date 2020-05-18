import cv2

scale_factor = 1.5
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    start = cv2.getTickCount()
    ret, frame = cap.read()
    dim = (int(frame.shape[1]*scale_factor), int(frame.shape[0]*scale_factor))
    new_img = cv2.resize(frame, dim)
    y = new_img.shape[0]
    x = new_img.shape[1]
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    # Display the resulting frame
    end = cv2.getTickCount()
    et = (end - start)/cv2.getTickFrequency()
    text = 'frame rate: {: .1}'.format(et)
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
    cv2.putText(new_img, text, (int(x * .01), int(y * .99)), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1)

    cv2.imshow('frame', new_img)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()