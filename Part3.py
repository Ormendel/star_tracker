import cv2
import numpy as np
import csv

img1 = cv2.imread('src/imageA.png', cv2.IMREAD_GRAYSCALE) # opening first image
img2 = cv2.imread('src/imageB.png', cv2.IMREAD_GRAYSCALE) # opening second image

sift = cv2.xfeatures2d.SIFT_create() # using SIFT as our algorithm for matching

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2) # K-nearest-neighbors with k = 2

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Save matched keypoints to CSV file
    with open('matched_keypoints.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'x', 'y', 'brightness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, m in enumerate(good_matches):
            if matches_mask[i]:
                kp = kp1[m.queryIdx]
                x, y = kp.pt
                brightness = kp.size
                writer.writerow({'id': i, 'x': x, 'y': y, 'brightness': brightness})
