# Wrinkle Detection Streamlit with YOLOv8

This project implements a Wrinkle Detection application using YOLOv8 for segmentations. The application is built with Streamlit and allows users to upload images for wrinkle detection of human faces.
There are 3 classes (forehead, frown and wrinkle) that the YOLOv8s (small) segmentation model was finetuned.

Dataset: [RoboFlow dataset](https://universe.roboflow.com/robbo/face-wrinkles-detection/dataset/1)


## Getting Started

### Prerequisites

- Python 3.11+
- pip
- opencv-contrib-python-headless
- ultralytics
- streamlit
- opencv-python

### Installation

Clone the repository:

```bash
git clone https://github.com/shrimantasatpati/Wrinkle-Detection-StreamLit.git
cd shrimantasatpati/Wrinkle-Detection-StreamLit
```

Finetuned model weight: [best.pt](best.pt)
