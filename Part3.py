import cv2
import numpy as np
import csv

img1 = cv2.imread(r".\src\fr1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r".\src\fr2.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

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
        fieldnames = ['id', 'x1', 'y1', 'r1', 'b1', 'x2', 'y2', 'r2', 'b2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        star_id = 1
        for i, m in enumerate(good_matches):
            if matches_mask[i]:
                kp_1 = kp1[m.queryIdx]
                (x1, y1) = kp_1.pt
                kp_2 = kp2[m.trainIdx]
                (x2, y2) = kp_2.pt
                radius_1 = 2.5 * kp_1.size
                radius_2 = 2.5 * kp_2.size
                brightness1 = kp_1.size
                brightness2 = kp_2.size
                writer.writerow({
                    'id': star_id,
                    'x1': round(x1, 4),
                    'y1': round(y1, 4),
                    'r1': round(radius_1, 4),
                    'b1': round(brightness1, 4),
                    'x2': round(x2, 4),
                    'y2': round(y2, 4),
                    'r2': round(radius_2, 4),
                    'b2': round(brightness2, 4)
                })
                star_id += 1
