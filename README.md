# Analog-Clock-Reader

## Project Overview
**Analog-Clock-Reader** is an image processing project aimed at reading the time from an analog clock image. This project leverages various image enhancement, edge detection, and feature extraction methods to detect the clock face, identify the clock hands, and accurately calculate the time.


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


You can install the required libraries using the provided `requirements.txt` file. Run the following command:

```bash
pip install -r requirements.txt
```


### Clone the Repository
```bash
git clone https://github.com/saugataroyarghya/Analog-Clock-Reader.git
cd Analog-Clock-Reader
```

## Usage
1. Run the Python script to star the application
    ```bash
    python main.py
    ```
2. Select an image of an analog clock in the `input_images` folder.
3. Maximize the window to see the output results with the intermediary steps

## Demo Images




## How it Works

1. **Preprocessing**:
    - The input image is converted to grayscale to simplify analysis, followed by gamma correction to enhance contrast.
    - Noise is reduced using median and bilateral filters, which preserve edges while smoothing out noise.
    - Adaptive thresholding is applied to convert the image to a binary format, making feature detection easier.

2. **Clock Face Detection**:
    - The largest contour in the binary image is identified and assumed to be the clock face based on its size and circular shape.

3. **Hand Detection**:
    - The Hough Line Transform is used to detect lines corresponding to the clock's hands.
    - Detected lines are classified as either hour or minute hands based on their length and angle.

4. **Time Calculation**:
    - The angles of the hour and minute hands are calculated relative to the 12 o'clock position.
    - These angles are then used to determine the corresponding time by mapping them to hour and minute values.

For more detailed information, refer to the [Report.pdf](./Report.pdf).



## Limitations
- Accuracy may be reduced if the clock image contains a second hand, has low contrast, or if the hands overlap.
- Complex backgrounds or distorted images may interfere with detection.

## Known Bugs
- Only 6 and 12 o clock is considered for the scenario where the hour hand and minute hand form a single line.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Acknowledgements
This project uses algorithms and image processing techniques including gamma correction, contour detection, Hough Line Transform, and morphological transformations. Special thanks to the creators and maintainers of the libraries used in this project.
