# Star Tracker
Second assignment in Intro to Space Engineering course.

In this project we implemented a star tracker.

##  submited by:
 
 * Or Mendel 315524389
 * Omer Michael 316334671
 * Eran Levy 311382360

![image](https://user-images.githubusercontent.com/57839539/232723922-003d2082-6fe0-4d75-b92e-fb1541bdc375.png)

### Note:

We've tried to change SIFT to ORB algorithm in order to get more matches.
We can't find the exact number of stars for each pair of images, but it works for most cases.

Here is the code part that didn't work well:

orb = cv2.ORB_create(nfeatures=10000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:100]

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

