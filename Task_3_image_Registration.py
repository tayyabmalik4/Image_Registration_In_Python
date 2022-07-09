import cv2
import numpy as np


img1 = cv2.imread("img1.jpg")  
img2 = cv2.imread("img3.jpg")  


img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


height, width = img2_gray.shape

orb_detector = cv2.ORB_create(5000)

kp1, d1_image = orb_detector.detectAndCompute(img1_gray, None)
kp2, d2_image = orb_detector.detectAndCompute(img2_gray, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = matcher.match(d1_image, d2_image)

matches = matches[:int(len(matches)*0.9)]

no_of_matches = len(matches)

p1_image = np.zeros((no_of_matches, 2))
p2_image = np.zeros((no_of_matches, 2))


for i in range(len(matches)):
  p1_image[i, :] = kp1[matches[i].queryIdx].pt
  p2_image[i, :] = kp2[matches[i].trainIdx].pt


  # Find the homography matrix.
homography, mask = cv2.findHomography(p1_image, p2_image, cv2.RANSAC)


# Use this matrix to transform the
# colored image wrt the reference image.
transformed_img = cv2.warpPerspective(img1,
                    homography, (width, height))

                    
cv2.imwrite('output.jpg', transformed_img)