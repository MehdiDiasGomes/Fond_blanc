# 🖼️ Image Processing Script

This script processes images by adjusting the white background and excluding images containing human elements. It uses OpenCV and NumPy libraries for image operations.

## Features ✨

1. **White Background Adjustment**: Adjusts the brightness of images so that the specified white point becomes pure white. 🎨
2. **Image Detection**: Checks if an image contains a sufficient proportion of white to be processed or not. 🔍
3. **Image Management**: Moves unprocessed images (containing human elements) to a specific folder. 📁

## Prerequisites 🛠️

Ensure you have the following libraries installed:
- `opencv-python`
- `numpy`

You can install them using pip:
```bash
pip install opencv-python numpy
