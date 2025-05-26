import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import csv


def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def image_gray(image,side='front'):
    #vrati plavu komponentu slike kao grayscale
    blue= image[:, :, 2]
    #obrada slike
    blue = cv2.GaussianBlur(blue, (5, 5), 0)
    blue_boosted = cv2.equalizeHist(blue)
    blue_boosted = np.uint8(np.absolute(blue_boosted))
    return blue
def image_bin(image_gs):
    return cv2.threshold(image_gs, 20,120, cv2.THRESH_BINARY)[1]
def image_bin_adaptive(image_gs):

    image_bin=cv2.adaptiveThreshold(image_gs, 120, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 401, 10)
    return image_bin
def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
def invert(image):
    return 255-image
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def find_largest_contour(image_bin):
    #traženje crepa
    contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def dilate(image_bin):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated_image = cv2.dilate(image_bin, kernel, iterations=1)
    return dilated_image
def erode(image_bin):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    eroded_image = cv2.erode(image_bin, kernel, iterations=1)
    return eroded_image

def load_dataset_images(folder_path):
    image_files = []
    image_files.extend(glob.glob(os.path.join(folder_path, '*.png')))
    return image_files

def isolate_tile(image, contour):
    #binarna maska
    mask = np.zeros_like(image[:, :, 0])  # Single channel mask for grayscale, or use [:, :, :3] for color  
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  
    isolated_tile = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(contour)
    isolated_tile = isolated_tile[y:y+h, x:x+w]
    #resize
    target_width, target_height = 1280, 960
    scale = min(target_width / w, target_height / h)
    new_size = (int(w * scale), int(h * scale))
    resized_tile = cv2.resize(isolated_tile, new_size, interpolation=cv2.INTER_AREA)
    #crna pozadina
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - resized_tile.shape[1]) // 2
    y_offset = (target_height - resized_tile.shape[0]) // 2
    canvas[y_offset:y_offset + resized_tile.shape[0], x_offset:x_offset + resized_tile.shape[1]] = resized_tile
    return canvas
def save_image(image, path):
    #pretvaranje slike u BGR format za OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)

#učitavanje referentnih slika
reference_image_path_front = './base_tiles/tile_front_larger.png'
reference_image_path_back = './base_tiles/tile_back_larger.png'
reference_image_front = load_image(reference_image_path_front)
reference_image_back = load_image(reference_image_path_back)
#rotacija referentnih slika radi lakšeg prepoznavanja
reference_image_front = rotate_image(reference_image_front, 90)
#reference_image_front=cv2.flip(reference_image_front, 1)
reference_image_back = rotate_image(reference_image_back, 90)
#reference_image_back = cv2.flip(reference_image_back, 1)
reference_image_front = cv2.resize(reference_image_front, (0, 0), fx=0.5, fy=0.5)
reference_image_back = cv2.resize(reference_image_back, (0, 0), fx=0.5, fy=0.5)
#učitavanje slika iz foldera
folder_path_front = './dataset/front/'
folder_path_back = './dataset/back/'
image_files_front = load_dataset_images(folder_path_front)
image_files_back = load_dataset_images(folder_path_back)


results=[]
#prolazak kroz sve slike u folderu
for image_path in image_files_front:
    im=load_image(image_path)
    #uveličavanje slike radi lakšeg prepoznavanja
    im_gray=image_gray(im)
    #pozadina je svetlija pa je potrebno invertovati sliku
    im_gray=invert(im_gray)

    im_bin=image_bin_adaptive(im_gray)
    im_bin=erode(im_bin)
    im_bin=dilate(im_bin)
    largest_contour_on_image=find_largest_contour(im_bin)
    if largest_contour_on_image is not None:
        #izolacija crepa
        isolated_tile = isolate_tile(im, largest_contour_on_image)
        save_image(isolated_tile, './isolated_tiles/front/' + os.path.basename(image_path))

for image_path in image_files_back:
    im=load_image(image_path)
    im_gray=image_gray(im)
    im_gray=invert(im_gray)
    im_bin=image_bin_adaptive(im_gray)
    largest_contour_on_image=find_largest_contour(im_bin)
    if largest_contour_on_image is not None:
        isolated_tile = isolate_tile(im, largest_contour_on_image)
        save_image(isolated_tile, './isolated_tiles/back/' + os.path.basename(image_path))

