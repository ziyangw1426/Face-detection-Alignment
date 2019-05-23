import cv2, dlib, argparse, os
import numpy as np
from mtcnn.mtcnn import MTCNN

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def align(input_image, face_dec):
    img = cv2.imread(input_image)
    if img is None: return None
    s_height, s_width = img.shape[:2]

    try:
        left_eye = detector_faces(img)[0]['left_eye']
        right_eye = detector_faces(img)[0]['right_eye']
    except:
        return None
    
    M = get_rotation_matrix(left_eye, right_eye)
    rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

    try:
        b = face_dec.detect_faces(rotated)[0]['box']
    except:
        return None
    
    if b != []:
        return crop_image_with_margin(rotated,b,0.3)
    else: 
        return None


def crop_image_with_margin(img_vector, b, margin):
    img_h, img_w, _ = np.shape(img_vector)
    x1, y1, w, h, x2, y2 = b[0], b[1], b[2], b[3], b[2] + b[0], b[3] + b[1]
    xw1 = max(int(x1 - margin * w), 0)
    yw1 = max(int(y1 - margin * h), 0)
    xw2 = min(int(x2 + margin * w), img_w - 1)
    yw2 = min(int(y2 + margin * h), img_h - 1)

    # width and height after applying factor
    new_w = xw2 - xw1 + 1
    new_h = yw2 - yw1 + 1
    
    # resize bounding box with shorter edge equal to required size
    required_size = 256
    shorter_edge = min(new_h, new_w)
    temp_img = img_vector[yw1:yw2 + 1, xw1:xw2 + 1, :]
    if shorter_edge == new_h:
        new_img = cv2.resize(temp_img, (required_size, int(required_size/new_w*new_h)))
        _, resize_h, _ = np.shape(new_img)
        residual = resize_h - required_size
        upper_margin = int(residual/2)
        lower_margin = resize_h - (residual - upper_margin)
        new_img = new_img[:,upper_margin:lower_margin,:]

    else:
        new_img = cv2.resize(temp_img, (required_size, int(required_size/new_w*new_h)))
        resize_w, _, _ = np.shape(new_img)
        residual = resize_w - required_size
        left_margin = int(residual/2)
        right_margin = resize_w - (residual - left_margin)
        new_img = new_img[left_margin:right_margin,:,:]
        
    new_img = cv2.resize(new_img, (required_size, required_size))
    return new_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align faces in image')
    parser.add_argument('input', type=str, help='')
    parser.add_argument('output', type=str, help='')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    
    face_dec = MTCNN()

    count = 0
    folder = os.listdir(args.input)
    size = len(folder)
    for i in folder:
        count += 1
        if count >= 100 and count % 100 == 0:
            print(str(count)+"/"+str(size)+" in progress...")
        res = align(args.input+"/"+i, face_dec)
        if res is not None:
            cv2.imwrite(args.output+"/%s.png"%count, res)
