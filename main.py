import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk



def apply_local_histogram_equalization_with_edges(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(cl, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(cl, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

    # Highlight the edges by adding the edge magnitude to the L-channel
    enhanced_l = cv2.addWeighted(cl, 0.9, sobel_mag, 0.1, 0)

    # Merge the enhanced L-channel with the a and b channels
    limg = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR color space
    equalized_image_with_edges = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return equalized_image_with_edges

def apply_gamma_correction(image, gamma = 1.0): 
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply the gamma correction using the lookup table
    corrected_image = cv2.LUT(image, table)
    return corrected_image

def remove_noise_(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding method
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small white regions
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Remove small black holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove small contours
    min_contour_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create a mask from the filtered contours
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=mask)

    return result_image

def remove_noise(image):
    # Use a series of median filters to reduce noise
    image = cv2.medianBlur(image, 3)
    image = cv2.medianBlur(image, 5)
    #image = cv2.medianBlur(image, 7)
    # Apply bilateral filter for further noise reduction
    image = cv2.bilateralFilter(image, 100, 2, 2)  # Adjust parameters as needed
    return image

def remove_noise_opening_(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding method
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small white regions
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Remove small black holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closing, opening

def removenoise_opening(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    return dilated_image, eroded_image

def closing(img, kernel_size=3):
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    closed_image = cv2.erode(dilated_image, kernel, iterations=2)
    return closed_image

def remove_small_components(img, min_size=10):
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Iterate through each component and remove small ones
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            img[labels == i] = 0
    return img

def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def increase_sharpness(image):
    # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply the sharpening kernel to the image
    sharp_image = cv2.filter2D(image, -1, kernel)

    return sharp_image

def find_angle(x1,y1,x2,y2):
    return np.degrees(np.arctan2(y2-y1, x2-x1))

def average_line(group):
    avg_x1 = np.mean([line[1][0] for line in group])
    avg_y1 = np.mean([line[1][1] for line in group])
    avg_x2 = np.mean([line[1][2] for line in group])
    avg_y2 = np.mean([line[1][3] for line in group])
    return (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2))

def distance_from_center(x1, y1, x2, y2, cx, cy):
    return abs((y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1) / math.sqrt((y2-y1)**2 + (x2-x1)**2)

def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)



def process_image(image_path):

    hour = 0
    minute = 0

    # Load the image 
    image = cv2.imread(image_path)
    #cv2.imwrite("image.jpg", image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray_image.jpg", gray_image)
    #gray_image = increase_sharpness(gray_image)
    #cv2.imwrite("sharp_image.jpg", gray_image)
    local_equalized_image_with_edges = apply_local_histogram_equalization_with_edges(image)
    
    cleaned_image = remove_noise(local_equalized_image_with_edges)
    #cv2.imwrite("cleaned_image.jpg", cleaned_image)
    gamma = 1.5
    gamma_corrected_image = apply_gamma_correction(cleaned_image, gamma)
    #cv2.imwrite("gamma_corrected_image.jpg", gamma_corrected_image)
    segmented_image_gray = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("segmented_image_gray.jpg", segmented_image_gray)
    
    blurred = cv2.GaussianBlur(segmented_image_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #cv2.imwrite("thresh.jpg", thresh)
    
    dilated_image, eroded_image = removenoise_opening(thresh)
    img_median = remove_noise(dilated_image)
    #cv2.imwrite("img_median.jpg", img_median)
    skeleton = morphology.skeletonize(img_median / 255)
    #cv2.imwrite("skeleton1.jpg", skeleton)
    skeleton = (skeleton*255).astype(np.uint8)
    #cv2.imwrite("skeleton2.jpg", skeleton)
    skeleton = closing(skeleton, 5)
    #cv2.imwrite("skeleton3.jpg", skeleton)
    skeleton = remove_small_components(skeleton, 100)
    #cv2.imwrite("skeleton4.jpg", skeleton)

    


    # Finding the clock and center of the clock
    contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Find the longest contour from the gray image and draw it on the gray image
    max_contour = max(contours, key = cv2.contourArea)
    cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
    #cv2.imwrite("contour.jpg", image)
    #Find the center of the longest contour
    M = cv2.moments(max_contour)
    highest_diameter = 0


    if M['m00'] == 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        leftmost = tuple(max_contour[max_contour[:,:,0].argmin()][0])
        rightmost = tuple(max_contour[max_contour[:,:,0].argmax()][0])
        topmost = tuple(max_contour[max_contour[:,:,1].argmin()][0])
        bottommost = tuple(max_contour[max_contour[:,:,1].argmax()][0])
        highest_diameter = max(distance(leftmost, rightmost), distance(topmost, bottommost))
    else:
        cx = image.shape[1] // 2
        cy = image.shape[0] // 2
        highest_diameter = image.shape[0] *5 // 6
    diameter_text = f"Highest diameter: {highest_diameter}"
    print(f"Highest diameter: {highest_diameter}")

    # Draw the center of the contour on the gray image with red color
    cv2.circle(image, (cx, cy), 10, (0,0,255), -1)
    #cv2.imwrite("center.jpg", image)
    center_text = f"Center: ({cx}, {cy})"
    print(f"Centerx: {cx} and Centery: {cy}")


    # Define parameters for Hough Line Transform
    rho = 1
    theta = np.pi / 180
    threshold = 10
    min_line_length = 50
    max_line_gap = 7

    # Copy of the image to draw lines on
    line_image = np.copy(image) * 0

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(skeleton, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines: 
        cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), 5)
    #cv2.imwrite("line_image.jpg", line_image)
    actual_lines = []

    # Filter lines based on proximity to the center
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    for line in lines:
        cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), 5)
        for x1, y1, x2, y2 in line:
            if (math.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2) < 60 or
                math.sqrt((x2 - center_x) ** 2 + (y2 - center_y) ** 2) < 60):
                actual_lines.append(line)
    #cv2.imwrite("actual_line_image.jpg", line_image)


    # Count the number of lines detected
    count = len(actual_lines)
    count1_text = f"Number of lines detected at the beginning: {count}"
    print(f"Number of lines detected at the beginning: {count}")


    if count <=1:
        if count ==1:
            for line in actual_lines:
                    for x1, y1, x2, y2 in line:
                        i = 255
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.line(line_image, (x1, y1), (x2, y2), (255 - i , i , i), 5)
            length = distance((actual_lines[0][0][0], actual_lines[0][0][1]), (actual_lines[0][0][2], actual_lines[0][0][3]))
            if length > int(highest_diameter//2):
                time_text = "The probable time is 6:00"
                hour_hand_text = "Hour hand: "
                minute_hand_text = "Minute hand: "
                twelve_clock_text = "12 o'clock position: "
                three_clock_text = "3 o'clock position: "
                hour_angle_text = "Hour hand angle: 0"
                minute_angle_text = "Minute hand angle: 180"
                hour = 6
                minute = 0
                time_text = "The time is 6:00"
                print("The time is 6:00")
            else: 
                hour_hand_text = "Hour hand: "
                minute_hand_text = "Minute hand: "
                twelve_clock_text = "12 o'clock position: "
                three_clock_text = "3 o'clock position: "
                hour_angle_text = "Hour hand angle: 0"
                minute_angle_text = "Minute hand angle: 0"
                hour = 12
                minute = 0
                time_text = "The time is 12:00"
                print("The time is 12:00")
            print()
        elif count == 0:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    d = distance_from_center(x1, y1, x2, y2, cx, cy)
                    if d < 60:
                        actual_lines.append(line)
            
            c = len(actual_lines)
            if c == 0:
                time_text = "The image is too complex to detect the clock"
                print("The image is too complex to detect the clock")
            elif c == 1:
                i=0
                for line in actual_lines:
                    for x1, y1, x2, y2 in line:
                        i = 255
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.line(line_image, (x1, y1), (x2, y2), (255 - i , i , i), 5)
                length = distance((actual_lines[0][0][0], actual_lines[0][0][1]), (actual_lines[0][0][2], actual_lines[0][0][3]))
                length_text = f"Length: {length}"
                print(f"Length: {length}")
                shape_text = f"Image shape: {image.shape[0]}"
                print(f"Image shape: {image.shape[0]}")
                if length > highest_diameter//2:
                    time_text = "The probable time is 6:00"
                    hour_hand_text = "Hour hand: "
                    minute_hand_text = "Minute hand: "
                    twelve_clock_text = "12 o'clock position: "
                    three_clock_text = "3 o'clock position: "
                    hour_angle_text = "Hour hand angle: 0"
                    minute_angle_text = "Minute hand angle: 180"
                    hour = 6
                    minute = 0
                    print("The probable time is 6:00")
                else:
                    hour_hand_text = "Hour hand: "
                    minute_hand_text = "Minute hand: "
                    twelve_clock_text = "12 o'clock position: "
                    three_clock_text = "3 o'clock position: "
                    hour_angle_text = "Hour hand angle: 0"
                    minute_angle_text = "Minute hand angle: 0"
                    hour = 6
                    minute = 0
                    time_text = "The probable time is 12:00"
                    print("The probable time is 12:00")

        else:
            time_text = "The image is too complex to detect the clock"
            print("The image is too complex to detect the clock")
            print()




    else:
    # Classify lines into hour and minute hands if more than 2 lines are detected
        if count > 2:
            slopes = []
            for line in actual_lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                    slopes.append((slope, line))

        # Sort the slopes
            slopes.sort(key=lambda x: x[0])

            # Cluster the slopes into two groups
            group1 = [slopes[0]]
            group2 = []
            for slope, line in slopes[1:]:
                if abs(slope - group1[0][0]) < 0.5:
                    group1.append((slope, line))
                else:
                    group2.append((slope, line))

            # Calculate the average line of each group
            hour_hand = np.mean([line for _, line in group1], axis=0)
            minute_hand = np.mean([line for _, line in group2], axis=0)

            actual_lines = [hour_hand, minute_hand]


        i = 0
        # Draw the lines
        for line in actual_lines:
            for x1, y1, x2, y2 in line:
                i = 255
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.line(line_image, (x1, y1), (x2, y2), (255 - i , i , i), 5)

        # Calculate the length of each line
        line_lengths = []
        for line in actual_lines:
            for x1, y1, x2, y2 in line:
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                line_lengths.append((length, line))

        # Sort the lines by length
        line_lengths.sort(key=lambda x: x[0])

        # Assign the shortest line to hour_hand and the second shortest to minute_hand
        hour_hand = line_lengths[0][1][0]
        minute_hand = line_lengths[1][1][0]

        # Ensure the coordinates are integers
        hour_hand = [int(coord) for coord in hour_hand]
        minute_hand = [int(coord) for coord in minute_hand]

        hour_hand_text = f"Hour hand: {hour_hand}"
        minute_hand_text = f"Minute hand: {minute_hand}"
        print("Hour hand:", hour_hand)
        print("Minute hand:", minute_hand)




        # Mathematicas for the time
        # Find the 12 o'clock position
        twelve_x = cx
        twelve_y = cy - 60

        # Find the 3 o'clock position

        three_x = cx + 60
        three_y = cy

        twelve_clock_text = f"12 o'clock position: ({twelve_x}, {twelve_y})"
        three_clock_text = f"3 o'clock position: ({three_x}, {three_y})"
        print(f"12 o'clock position: ({twelve_x}, {twelve_y})")
        print(f"3 o'clock position: ({three_x}, {three_y})")


        hour_angle_12 = angle(
            np.array([twelve_x, twelve_y], dtype=np.float64),
            np.array([cx, cy], dtype=np.float64),
            np.array([((hour_hand[2]+hour_hand[0])/2),((hour_hand[1]+hour_hand[3])/2) ], dtype=np.float64)
        )
        hour_angle_3 = angle(
            np.array([three_x, three_y], dtype=np.float64),
            np.array([cx, cy], dtype=np.float64),
            np.array([hour_hand[2], hour_hand[3]], dtype=np.float64)
        )

        minute_angle_12 = angle(
            np.array([twelve_x, twelve_y], dtype=np.float64),
            np.array([cx, cy], dtype=np.float64),
            np.array([((minute_hand[0]+minute_hand[2])/2), ((minute_hand[1]+minute_hand[3])/2)], dtype=np.float64)
        )

        minute_angle_3 = angle(
            np.array([three_x, three_y], dtype=np.float64),
            np.array([cx, cy], dtype=np.float64),
            np.array([minute_hand[2], minute_hand[3]], dtype=np.float64)
        )

        if hour_angle_3 > 90: 
            hour_angle = 360 - hour_angle_12
        else:
            hour_angle = hour_angle_12

        if minute_angle_3 > 90:
            minute_angle = 360 - minute_angle_12
        else:
            minute_angle = minute_angle_12

        hour_angle_text = f"Hour hand angle: {hour_angle}"
        minute_angle_text = f"Minute hand angle: {minute_angle}"
        print(f"Hour hand angle: {hour_angle}")
        print(f"Minute hand angle: {minute_angle}")

        minute = int(minute_angle // 6)  # Each minute is 6 degrees
        hour = int(hour_angle // 30)  # Each hour is 30 degrees
        if hour == 0:
            hour = 12
        time_text = f"The probable time is: {hour} hours and {minute} minutes"
        print(f"The probable time is: {hour} hours and {minute} minutes")




    # Count the number of lines detected
    count = len(actual_lines)
    count_text = f"Number of lines detected: {count}"
    print(f"Number of lines detected: {count}")





    # Display results
    plt.figure(figsize=(14, 16))

    plt.subplot(2, 4, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #plt.subplot(3, 4, 2)
    #plt.title('Local Histogram Equalized Image with Edges')
    #plt.imshow(cv2.cvtColor(local_equalized_image_with_edges, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 4, 2)
    plt.title('Gamma Corrected Image')
    plt.imshow(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 4, 3)
    plt.title('Blurred Image')
    plt.imshow(blurred, cmap='gray')

    plt.subplot(2, 4, 4)
    plt.title('Adaptive Thresholding')
    plt.imshow(thresh, cmap='gray')

    #plt.subplot(3, 4, 6)
    #plt.title('Eroded Image')
    #plt.imshow(eroded_image, cmap='gray')

    plt.subplot(2, 4, 5)
    plt.title('Dilated Image')
    plt.imshow(dilated_image, cmap='gray')

    plt.subplot(2, 4, 6)
    plt.title('After Median Filter')
    plt.imshow(img_median, cmap='gray')

    plt.subplot(2, 4, 7)
    plt.title('Skeletonization')
    plt.imshow(skeleton, cmap='gray')

    plt.subplot(2, 4, 8)
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    plt.imshow(cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines')
    plt.axis('off')

    # top=0.952,
    # bottom=0.052,
    # left=0.01,
    # right=0.99,
    # hspace=0.321,
    # wspace=0.041
    plt.tight_layout(rect=[0.01,0.052,0.99,0.952])
    plt.subplots_adjust(hspace=0.321, wspace=0.041)
    plt.show()



    cv2.circle(image, (cx, cy), 3, (0,0,255), -1)
    image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    # cv2.imshow('Center', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    output_image = image
    #time_text = f"The probable time is: {hour} hours and {minute} minutes"
    return output_image, time_text, count_text, count1_text, hour_hand_text, minute_hand_text, twelve_clock_text, three_clock_text, hour_angle_text, minute_angle_text, diameter_text, center_text
def display_image(image, label):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.thumbnail((400, 400))
    image = ImageTk.PhotoImage(image)
    label.config(image=image)
    label.image = image
def open_file():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", ".jpg .jpeg .png .bmp .webp")])
    if file_path:
        original_image = cv2.imread(file_path)
        processed_image, result_text, count_text, count1_text, hour_hand_text, minute_hand_text, twelve_clock_text, three_clock_text, hour_angle_text, minute_angle_text, diameter_text, center_text = process_image(file_path)
        #processed_image, result_text,  = process_image(file_path)
        display_image(original_image, original_image_label)
        display_image(processed_image, processed_image_label)
        result_label.config(text=result_text)
        count_label.config(text=count_text)
        count1_label.config(text=count1_text)
        hour_hand_label.config(text=hour_hand_text)
        minute_hand_label.config(text=minute_hand_text)
        twelve_clock_label.config(text=twelve_clock_text)
        three_clock_label.config(text=three_clock_text)
        hour_angle_label.config(text=hour_angle_text)
        minute_angle_label.config(text=minute_angle_text)
        diameter_label.config(text=diameter_text)
        center_label.config(text=center_text)
        




root = tk.Tk()
root.title("Clock Time Detection")

# Increase window size
root.geometry("800x600")

original_image_frame = tk.Frame(root)
original_image_frame.pack(side="left", padx=10, pady=10)
original_image_label = tk.Label(original_image_frame)
original_image_label.pack()
#original_image_label.config(width=200, height=200)  # Set the width and height to 200 pixels

processed_image_frame = tk.Frame(root)
processed_image_frame.pack(side="left", padx=10, pady=10)
processed_image_label = tk.Label(processed_image_frame)
processed_image_label.pack()
#processed_image_label.config(width=200, height=200)  # Set the width and height to 200 pixels


result_label = tk.Label(root, text="Select an image to process", font=("Arial", 16))
result_label.pack(pady=20)
open_button = tk.Button(root, text="Open Image", command=open_file)
open_button.pack(pady=10)

count_label = tk.Label(root, text="", font=("Arial", 16))
count_label.pack(pady=10)
count1_label = tk.Label(root, text="", font=("Arial", 16))
count1_label.pack(pady=10)
hour_hand_label = tk.Label(root, text="", font=("Arial", 16))
hour_hand_label.pack(pady=10)
minute_hand_label = tk.Label(root, text="", font=("Arial", 16))
minute_hand_label.pack(pady=10)
twelve_clock_label = tk.Label(root, text="", font=("Arial", 16))
twelve_clock_label.pack(pady=10)
three_clock_label = tk.Label(root, text="", font=("Arial", 16))
three_clock_label.pack(pady=10)
hour_angle_label = tk.Label(root, text="", font=("Arial", 16))
hour_angle_label.pack(pady=10)
minute_angle_label = tk.Label(root, text="", font=("Arial", 16))
minute_angle_label.pack(pady=10)
diameter_label = tk.Label(root, text="", font=("Arial", 16))
diameter_label.pack(pady=10)
center_label = tk.Label(root, text="", font=("Arial", 16))
center_label.pack(pady=10)



root.mainloop()