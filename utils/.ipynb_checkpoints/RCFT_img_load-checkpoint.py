import numpy as np
import cv2

def load_img(filename, image_shape):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if image.shape[0] > image.shape[1]:
        p = image.shape[0] - image.shape[1]
        img_pad = np.pad(image, ((0, 0), (int(p / 2), int(p / 2))), 'constant', constant_values=0)

    else:
        p = image.shape[1] - image.shape[0]
        img_pad = np.pad(image, ((int(p / 2), int(p / 2)), (0, 0)), 'constant', constant_values=0)


    img_pad = cv2.resize(img_pad, (image_shape, image_shape))

    img_pad_norm = cv2.normalize(img_pad, None, 0, 1, cv2.NORM_MINMAX)

    img_pad_norm_color = cv2.cvtColor(img_pad_norm, cv2.COLOR_GRAY2RGB)
    img_pad_norm_color = np.moveaxis(img_pad_norm_color, -1, 0)

    return img_pad_norm_color