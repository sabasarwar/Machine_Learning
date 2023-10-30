import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("Test_02.jpg")

# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))  # -1 reshape means, in this case MxN

# Convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
img2 = np.float32(img2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Number of clusters
k = 10

# Number of attempts, number of times algorithm is executed using different initial labeling
# Algorithm return labels that yield the best compactness.
# compactness : It is the sum of squared distance from each point to their corresponding centers.

attempts = 10

# other flags needed as inputs for K-means
# Specify how initial seeds are taken.
# Two options, cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS

compactness,label,center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
# cv2.kmeans outputs 2 parameters.
# 1 Compactness.
# 2 Labels: Label array.
# 3 Center. the array of centers of clusters. For k=4 we will have 4 centers.
# For RGB image, we will have center for each image, so total 4x3 = 12.
# Now convert center values from float32 back into uint8.
center = np.uint8(center)

# Next, we have to access the labels to regenerate the clustered image
res = center[label.flatten()]
res2 = res.reshape((img.shape)) # Reshape labels to the size of original image
cv2.imwrite("segmented_image_02.jpg", res2)
