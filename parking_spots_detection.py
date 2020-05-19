import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np


# converts a BGR image to a Grayscale image
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Keeps only the pixels with a specific white value
def filter_white_mask(image):
    lower = np.uint8([130, 150, 140])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    return white_mask


# Apply the fastNlMeansDenoising algorithm to a given image
def de_noise(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 3, 15)


# Checks if the given line segment intersects with at least one
# of the line segments from the given list
def edge_is_valid(edges, x1, y1, x2, y2):
    for edge in edges:
        for edge_x1, edge_y1, edge_x2, edge_y2 in edge:
            p1 = Point(edge_x1, edge_y1, )
            q1 = Point(edge_x2, edge_y2)
            p2 = Point(x1, y1)
            q2 = Point(x2, y2)
            if do_intersect(p1, q1, p2, q2):
                return True
    return False


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# point q lies on line segment 'pr'
def on_segment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:

        # Clockwise orientation
        return 1
    elif val < 0:

        # Counterclockwise orientation
        return 2
    else:

        # Collinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def do_intersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True

    # If none of the cases
    return False


root = tk.Tk()
root.withdraw()

# open file dialog and display the selected image
file_path = filedialog.askopenfilename(title="Choose an image", filetypes=[('image files', ('.png', '.jpg'))])
input_image = cv2.imread(file_path)
input_aux = input_image.copy()
# show image
cv2.imshow('Input', input_image)

# Applying Gaussian blur to the input_image
blurred = cv2.GaussianBlur(input_image, (3, 3), 0)

# De-noising the image
de_noised = de_noise(blurred)
de_noised_twice = de_noise(de_noised)

# Keeping only the wanted white pixels
filtered_white = filter_white_mask(de_noised_twice)

# Finding the longest lines from the parking, in this case the horizontal ones
main_lines = cv2.HoughLinesP(image=filtered_white, rho=1, theta=np.pi / 10, threshold=120, minLineLength=300,
                             maxLineGap=3)

# Drawing the found lines
for line in main_lines:
    for x1, y1, x2, y2 in line:
        cv2.line(input_image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

# Finding the vertical lines
vertical_lines = cv2.HoughLinesP(image=filtered_white, rho=1, theta=np.pi, threshold=1, minLineLength=60,
                                 maxLineGap=3)

# Drawing the found lines if they intersect with the already drown ones
for line in vertical_lines:
    for x1, y1, x2, y2 in line:
        if edge_is_valid(main_lines, x1, y1, x2, y2):
            cv2.line(input_image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

# Start parking contour detection
multiple_lines = cv2.HoughLinesP(image=filtered_white, rho=1, theta=np.pi / 180, threshold=145, minLineLength=20,
                                 maxLineGap=35)
# Accentuate the found lines on an input image copy
for i in range(len(multiple_lines)):
    for x1, y1, x2, y2 in multiple_lines[i]:
        cv2.line(input_aux, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)

# Process the obtained image
gray = convert_to_grayscale(input_aux)
blur = cv2.medianBlur(gray, 5)
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
thresh = cv2.threshold(sharpen, 10, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

# Apply the contour finding algorithm
contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Make a copy to overlay the resulted rectangles
overlay = input_image.copy()

# Filter the found contours to keep only the largest ones
# Apply them to the output image
min_area = 7000
max_area = 90000
image_number = 0
for c in contours:
    area = cv2.contourArea(c)
    if min_area < area < max_area:
        x2, y2, w2, h2 = cv2.boundingRect(c)
        ROI = overlay[y2:y2 + h2, x2:x2 + h2]
        cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv2.rectangle(overlay, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), -1)
        image_number += 1

# apply the overlay to the final image
alpha = 0.2
cv2.addWeighted(src1=overlay, alpha=alpha, src2=input_image, beta=1 - alpha, gamma=0, dst=input_image)

# Displaying the final output image
cv2.imshow('Output', input_image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
