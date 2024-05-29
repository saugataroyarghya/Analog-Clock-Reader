import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('img5.jpg')

# Convert the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1, minDist=120, param1=100, param2=100, minRadius=40, maxRadius=0
)

# If some circles are detected, let's draw them
if circles is not None:
    circles = np.uint16(np.around(circles))  # Convert the (x, y, radius) values to integers
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# Display the image with detected circles using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Circles')
plt.axis('off')
plt.show()


kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 50
max_line_gap = 2
line_image = np.copy(image) * 0

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)

#SHow the image with the lines
plt.imshow(cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.axis('off')
plt.show()
