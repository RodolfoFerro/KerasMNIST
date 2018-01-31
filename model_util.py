from keras.models import model_from_json
import numpy as np
import cv2


def load_model():
    # Open model from JSON:
    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()

    # Load model:
    model = model_from_json(model_json)

    # Load weights into model:
    model.load_weights("model.h5")
    print("Loaded model from disk.")

    # Compile model:
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model._make_predict_function()
    return model


def process_img(input_img):
    # Read the input image
    im = cv2.imread("static/img/" + input_img)

    # Resize image if necessary:
    if im.shape[1] > 3000 or im.shape[0] > 3000:
        im = cv2.resize(im, (im.shape[1]//8, im.shape[0]//8))
    elif im.shape[1] > 2000 or im.shape[0] > 2000:
            im = cv2.resize(im, (im.shape[1]//4, im.shape[0]//4))
    elif im.shape[1] > 1000 or im.shape[0] > 1000:
        im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image:
    im2, ctrs, hier = cv2.findContours(im_th.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour:
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    return im, im_th, rects


def predict(model, im, im_th, rects, output_img):
    # Results:
    res = []

    # For each rectangular region, predict the digit using CNN model:
    for rect in sorted(rects):
        # Draw the rectangles:
        cv2.rectangle(im, (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]),
                      (10, 255, 180), 3)

        # Make the rectangular region around the digit:
        leng = int(rect[3] * 1.2)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

        # Resize the image:
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        # Reshape to be [samples][pixels][width][height]:
        X = roi.reshape(1, 1, 28, 28).astype('float32')
        X /= 255

        # Predict digit:
        nbr = np.argmax(model.predict(X))
        res.append(nbr)
        cv2.putText(im, str(nbr), (rect[0], rect[1]),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (50, 190, 255), 3)

    # Save results:
    cv2.imwrite("static/img/" + output_img, im)
    return ''.join(map(str, res))
