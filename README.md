# Analog-Clock-Reader

## Project Overview
**Analog-Clock-Reader** is a computer vision project aimed at reading the time from an analog clock image using image processing techniques. This project leverages various image enhancement, edge detection, and feature extraction methods to detect the clock face, identify the clock hands, and accurately calculate the time.

## Objectives
- Enhance the quality of input images for better analysis.
- Detect the clock face and identify the clock hands.
- Calculate and display the time based on the positions of the detected hands.

## Features
- **Preprocessing**: Applies gamma correction, noise removal, and thresholding to improve image clarity.
- **Clock Face Detection**: Detects and highlights the clock face using contour analysis.
- **Hand Detection**: Identifies clock hands using the Hough Line Transform and filters lines based on their proximity to the clock center.
- **Time Calculation**: Computes the time based on the angles of the hour and minute hands relative to the 12 o'clock position.

## Installation

### Prerequisites
Ensure that you have the following libraries installed:
- [OpenCV](https://opencv.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Skimage](https://scikit-image.org/)
- [Math](https://docs.python.org/3/library/math.html)
- [Tkinter](https://wiki.python.org/moin/TkInter)
- [PIL](https://pillow.readthedocs.io/)

You can install them using pip:
```bash
pip install opencv-python numpy matplotlib scikit-image pillow
```

### Clone the Repository
```bash
git clone https://github.com/saugataroyarghya/Analog-Clock-Reader.git
cd Analog-Clock-Reader
```

## Usage
1. Place the image of an analog clock in the `input_images` folder.
2. Run the Python script to process the image:
    ```bash
    python clock_reader.py --image input_images/your_clock_image.jpg
    ```
3. The processed image, along with the detected clock face, hands, and calculated time, will be displayed.

## How it Works
1. **Preprocessing**:
    - The input image is first converted to grayscale and enhanced using gamma correction.
    - Noise is reduced using median and bilateral filtering.
    - Adaptive thresholding is applied to convert the image to binary for easier feature detection.
  
2. **Clock Face Detection**:
    - The largest contour in the binary image is detected and assumed to be the clock face.
  
3. **Hand Detection**:
    - The Hough Line Transform is used to detect lines representing the clock hands.
    - Detected lines are classified as either hour or minute hands based on their lengths and angles.
  
4. **Time Calculation**:
    - The angles of the hour and minute hands are calculated relative to the 12 o'clock position, and the corresponding time is determined.

## Limitations
- Accuracy may be reduced if the clock image contains a second hand, has low contrast, or if the hands overlap.
- Complex backgrounds or distorted images may interfere with detection.
  
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Acknowledgements
This project uses algorithms and image processing techniques including gamma correction, contour detection, Hough Line Transform, and morphological transformations. Special thanks to the creators and maintainers of the libraries used in this project.
