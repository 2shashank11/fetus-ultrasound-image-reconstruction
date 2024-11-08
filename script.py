import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

image_dir = './dataset/images'

df = pd.read_csv('./dataset/image_label.csv')
femur = df[df['Plane']=='Fetal femur'].iloc[16:]

def display_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def load_images(directory):
    images = []
    for filename in femur['Image_name']:
        img_path = os.path.join(directory, filename + '.png')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        if image is not None:
            images.append(image)
    return images

def resizeImage(img):
    target_size = (512, 512)
    old_size = img.shape[:2]
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
    img_resized = cv2.resize(img, new_size)
    
    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img_padded

def preprocess_image(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

def segment_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    return binary_img

def apply_convex_hull(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hull_img = np.zeros_like(image)

    if contours:
        longest_distance = 0
        longest_contour = None
        for contour in contours:
            for i in range(len(contour)):
                for j in range(i + 1, len(contour)):
                    dist = cv2.norm(contour[i][0] - contour[j][0])
                    if dist > longest_distance:
                        longest_distance = dist
                        longest_contour = contour

        if longest_contour is not None:
            hull = cv2.convexHull(longest_contour)
            cv2.drawContours(convex_hull_img, [hull], -1, 255, thickness=cv2.FILLED)
    return contours, convex_hull_img

def extract_femur_length(image, original_image):
    marked_image = original_image.copy()
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_farthest_dist = 0
    point1, point2 = None, None
    
    for contour in contours:
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                dist = cv2.norm(contour[i][0] - contour[j][0])
                if dist > max_farthest_dist:
                    max_farthest_dist = dist
                    point1, point2 = tuple(contour[i][0]), tuple(contour[j][0])

    if point1 and point2:
        cv2.circle(marked_image, point1, 8, (0, 255, 0), -1)
        cv2.circle(marked_image, point2, 8, (0, 0, 255), -1)
        cv2.line(marked_image, point1, point2, (255, 0, 0), 3)
    
    marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
    return max_farthest_dist, point1, point2, marked_image

def fetus_height(femur_length):
    image_dpi = 96
    femur_length_in_cm = ((femur_length/image_dpi)*2.54)
    fetus_height = 6.18 + 0.59*femur_length_in_cm*10
    return fetus_height, femur_length_in_cm
