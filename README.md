# BYU Locating Bacterial Flagellar Motors - YOLOv8 Implementation

## Overview
This project is a solution for the [BYU Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025) competition. The objective is to identify the 3D coordinates $(x, y, z)$ of flagellar motors within electron cryotomography (cryo-ET) data.

This notebook implements a **2D slice-based approach** using **YOLOv8 (You Only Look Once)**. Instead of using native 3D object detection, we treat the tomogram slices as a video stream or independent images, detect the motors in 2D, and then aggregate the results to form a 3D prediction.

## Key Features
* **Offline Installation**: Includes logic to install `ultralytics` from local wheel files, complying with Kaggle code competition constraints where internet access is disabled.
* **3D to 2D Conversion**: Converts 3D tomogram arrays into normalized 2D JPEG images for YOLO training.
* **Contrast Enhancement**: Uses percentile-based normalization (2nd to 98th percentile) to handle the high dynamic range of cryo-ET data.
* **Custom 3D NMS**: Implements a 3D Non-Maximum Suppression algorithm to merge detections across multiple Z-slices into a single confident 3D coordinate.
* **Visualization**: Extensive EDA including 3D scatter plots of motor distributions and overlays of bounding boxes on training data.

## Methodology

### 1. Data Preprocessing
The raw data consists of 3D numpy arrays. The model cannot ingest these directly.
* **Slice Extraction**: The script iterates through the training labels. For every ground truth motor, it extracts the central Z-slice and `±6` neighboring slices (defined by `TRUST = 6`).
* **Normalization**: Raw voxel values are clipped between the 2nd and 98th percentiles and scaled to `0-255` (uint8).
* **Label Generation**: YOLO formatted `.txt` files are generated containing the class ID (0) and the normalized bounding box coordinates.
* **Dataset Split**: 80% Training / 20% Validation.

### 2. Model Architecture
* **Framework**: Ultralytics YOLOv8.
* **Variant**: `yolov8n.pt` (Nano) - chosen for speed and efficiency.
* **Training Config**:
    * Image Size: 640px
    * Epochs: 30
    * Batch Size: 16
    * Patience: 5 epochs

### 3. Inference & Post-Processing
During inference on the test set:
1.  **Batch Processing**: The notebook processes test tomograms slice-by-slice using a custom `GPUProfiler` and threading to optimize data loading.
2.  **Detection**: YOLO runs on the slices to find potential motor candidates.
3.  **3D Aggregation**: A custom `perform_3d_nms` function calculates the Euclidean distance between detections in 3D space. Detections falling within the motor radius are merged, and the one with the highest confidence is selected.

## Dependencies
The notebook requires the following Python libraries:
* `ultralytics` (YOLOv8)
* `torch` / `torchvision`
* `numpy`
* `pandas`
* `opencv-python` (cv2)
* `Pillow` (PIL)
* `matplotlib` / `seaborn` (for visualization)
* `tqdm` (progress tracking)

## Directory Structure
The notebook expects the Kaggle competition data structure:

```text
/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/
├── train/              # Directories of training tomograms
├── test/               # Directories of test tomograms
├── train_labels.csv    # CSV containing target coordinates
└── sample_submission.csv

The notebook generates a working directory for training:

/kaggle/working/
├── yolo_dataset/       # Generated images and labels
├── yolo_weights/       # Trained model weights
└── submission.csv      # Final inference results
```
## Setup and Usage
1. Environment Setup: Ensure GPU acceleration is enabled. The notebook automatically detects if CUDA is available.


```
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
2. Data Loading: The notebook automatically creates the yolo_dataset directory structure. Run the prepare_yolo_dataset() function cell to process the raw tomograms into images.

3. Training: Run the train_yolo_model() function. This will fine-tune the YOLOv8 model and save the best weights to yolo_weights/motor_detector/weights/best.pt.

4. Inference: The inference block iterates through the /test folder.

  - Adjust CONCENTRATION (default 1) to skip slices if inference is too slow (e.g., 0.5 processes every other slice).

  - Adjust CONFIDENCE_THRESHOLD (default 0.45) to filter weak predictions.

## Results
- Loss Curves: The notebook saves dfl_loss_curve.png to visualize training progress.

- Submission: Generates submission.csv with columns: tomo_id, Motor axis 0 (z), Motor axis 1 (y), Motor axis 2 (x).
- [Link to kaggle notebook](https://www.kaggle.com/code/jemimaegwurube/byu-submission-notebook) 
