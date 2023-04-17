import warnings
import cv2
import numpy as np
import random
import math
import csv

# Load the image
img = cv2.imread(r".\src\fr2.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply adaptive threshold to the image to isolate stars
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define the RANSAC parameters
num_iterations = 1000
threshold_distance = 5
num_points_for_model = 2 # this is how RANSAC works, by 2 points

# Define a function to calculate the distance between a point and a line
def point_line_distance(point, line):
    x0, y0 = point
    a, b, c = line
    return abs(a*x0 + b*y0 + c) / math.sqrt(a*a + b*b)

# Define a function to fit a line to a set of points using the least squares method
def fit_line(points):
    x = points[:,0]
    y = points[:,1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, -1, b

# Loop over the contours and extract the coordinates of the stars
stars = []
for cnt in contours:
    # Compute the center and radius of the contour
    (x, y), r = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))

    # Extract the brightness of the star
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            star_region = gray[int(y) - 5:int(y) + 6, int(x) - 5:int(x) + 6]
            if star_region.size > 0 and np.std(star_region) > 1:
                b = int(np.mean(star_region))
            else:
                b = -1
        except ValueError:
            b = -1

    # Add the coordinates and brightness to the list of stars
    stars.append((center[0], center[1], r , b))

# Perform RANSAC to find the coordinates of the stars
stars_cord = []
for i in range(num_iterations):
    # Randomly select a set of points to fit a line
    points = random.sample(stars, num_points_for_model)
    line = fit_line(np.array(points))

    # Count the number of stars within the threshold distance of the line
    num_inliers = 0
    inliers = []
    for star in stars:
        distance = point_line_distance((star[0], star[1]), line)
        if distance < threshold_distance:
            num_inliers += 1
            inliers.append(star)

    # If the current set of points produces more inliers than any previous set, update the inliers list
    if num_inliers > len(stars_cord):
        stars_cord = inliers

# Save the coordinates of the stars to a CSV file
with open('stars_cord.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'x', 'y', 'r', 'b'])
    for i, star in enumerate(stars_cord):
        writer.writerow([i+1, round(star[0],4), round(star[1],4), round(star[2],4), round(star[3],4)])
