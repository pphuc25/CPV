import cv2
import os

save_path = 'D:\FPT\SPRING23\CPV301\CPV301_code\CPV\workshop8\Dataset_resize'
os.makedirs(save_path, exist_ok=True)

path = 'D:\FPT\SPRING23\CPV301\CPV301_code\CPV\workshop8\Dataset'
image_names = os.listdir(path)
dim = (64, 64)

for i in image_names:
    img = cv2.imread(os.path.join(path, i), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    image_path = os.path.join(save_path, f'{i}')
    cv2.imwrite(image_path, resized)
