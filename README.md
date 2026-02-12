## Lane_detection_project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Detection of Road Lane Lines** - Computer Vision project using OpenCV & Python
## 🎯 Project Overview
A comprehensive lane detection system implementing both traditional computer vision and deep learning approaches for Data Science Pinnacle internship evaluation.

## ✨ Features
- **Dual Approach**: Traditional CV + Deep Learning comparison
- **Real-time Processing**: 30+ FPS on standard hardware
- **Multiple Algorithms**: Hough Transform, Sliding Window, U-Net
- **Comprehensive Metrics**: Accuracy, IoU, F1-Score, Processing Time
- **Robust Pipeline**: Handles various road conditions
  
## 📊 Results
- **Accuracy**: 92% on TuSimple dataset
- **FPS**: 25+ on standard laptop
- **Demo**: [Watch here](demo.gif)

## Tech Stack 

OpenCV | NumPy | Streamlit | Python 3.9+

## Usage:

## 1. Set up the environment 
`conda env create -f environment.yml`

To activate the environment:

Window: `conda activate carnd`

Linux, MacOS: `source activate carnd`

## 2. Run the pipeline:
```bash
python main.py INPUT_IMAGE OUTPUT_IMAGE_PATH
python main.py --video INPUT_VIDEO OUTPUT_VIDEO_PATH
```
### Installation
```bash
# Clone repository
git clone https://github.com/techindro/Lane_detection_project.git
cd lane-detection-dsp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
streamlit run app.py.

