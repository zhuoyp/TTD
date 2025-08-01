# The Trauma THOMPSON Dataset

## How to Access the Dataset

To ensure responsible use, access to the dataset is granted after agreeing to our terms of use.

Fill out the Consent Form: Please complete the user consent form at the following [link](https://docs.google.com/forms/d/e/1FAIpQLSeWBzYkKBzAS5U1TwFMZDfauEgXvioega9-qtuXDbzJQ5glTQ/viewform?usp=dialog). 

Dataset Access Registration Form  (<- Replace with your actual Google Form link)

Receive Access Links: Upon submission of the form, you will be automatically redirected to a page containing the download links for the dataset.

Download the Data: The dataset is hosted on both Google Drive and Harvard Dataverse for reliable access.

Google Drive: A private link will be provided after consent.

Harvard Dataverse: For permanent and version-controlled access.


## 📦 Dataset Description

The **Trauma THOMPSON Dataset** is designed for multimodal research in egocentric video analysis of life-saving medical procedures. It supports multiple tasks including action recognition, hand tracking, object detection, and visual question answering.

---

### 🔍 Action Recognition & Anticipation
- **177** videos of standard (regular) emergency procedures  
- **43** videos of *just-in-time* (JIT) procedures  

---

### ✋ Hand Tracking
- **30** videos (subset of the regular procedure dataset)  
- Annotations include:
  - Hand bounding boxes in COCO format
  - Left/right hand identification

---

### 🛠 Object Detection
- **25,000** frames  
- **12** common surgical tools
- Bounding box annotations for tool presence and location in YOLO format

---

### ❓ Visual Question Answering (VQA)
- **600,000** frames with corresponding VQA annotations  

---

## 📁 Dataset Structure  

### **1. Action Recognition & Anticipation**  
**Input:** Pre-extracted video frames  
**Folder Structure:**  
```
actions/  
├── videos/  
│   ├── P01_01_00/  
│   │   ├── img_00001.jpg  
│   │   ├── img_00002.jpg  
│   ├── P01_01_01/  
│   │   ├── img_00001.jpg  
│   │   ├── img_00002.jpg  
│   ├── ...  
├── annotations.csv  
```  
**Annotations:**  
- `annotations.csv` contains video-level labels (e.g., procedure type, timestamps).  

---

### **2. Hand Tracking**  
**Input:** Full videos + per-frame bounding boxes  
**Folder Structure:**  
```
hands/  
├── videos/  
│   ├── P01_01.mp4  
│   ├── P01_02.mp4  
│   ├── ...  
├── bbx/  
│   ├── P01_01/  
│   │   ├── P01_01_00001.json  
│   │   ├── P01_01_00002.json  
│   ├── P01_02/  
│   │   ├── P01_02_00001.json  
│   │   ├── P01_02_00002.json  
│   ├── ...  
```  
**Annotation Format (COCO-style JSON):**  
```json
{
    "categories": [
        {"id": 0, "name": "left hand", "supercategory": "hand"},
        {"id": 1, "name": "right hand", "supercategory": "hand"}
    ],
    "images": [
        {
            "id": frame_number,
            "file_name": "P01_01_00001.jpg",
            "height": h,
            "width": w,
            "channel": 3
        }
    ],
    "annotations": [
        {
            "id": annotation_id,
            "image_id": frame_number,
            "category_id": 0,  # 0=left, 1=right
            "bbox": [x, y, w, h],  # COCO format (x,y top-left)
            "area": w * h,
            "iscrowd": 0
        }
    ]
}
```

---

### **3. Object Detection (Surgical Tools)**  
**Input:** Single frames + tool annotations  
**Folder Structure:**  
```
objects/  
├── images/  
│   ├── P01_01_00001.jpg  
│   ├── P01_01_00002.jpg  
│   ├── ...  
├── labels/  
│   ├── P01_01_00001.json  
│   ├── P01_01_00002.json  
│   ├── ...  
```  
**Annotation Format (YOLO-style):**  
- Each `.json` file contains:  
  ```json
  {
      "object-class": 0,  # Class ID (0-11 for 12 tools)
      "bbox": [x_center, y_center, width, height]  # Normalized [0,1]
  }
  ```

---

### **4. Visual Question Answering (VQA)**  
**Input:** Frames + question-answer pairs  
**Folder Structure:**  
```
vqa/  
├── images/  
│   ├── P01_01_00001.jpg  
│   ├── P01_01_00002.jpg  
│   ├── ...  
├── questions.json  
├── annotations.json  
```  

**Question Format (`questions.json`):**  
```json
{
    "info": {
        "description": "Trauma THOMPSON VQA dataset",
        "version": "1.0",
        "year": 2025
    },
    "questions": [
        {
            "image_id": "P01_01_00001.jpg",
            "question": "What tool is in the surgeon's right hand?",
            "question_id": 1
        }
    ]
}
```

**Answer Format (`annotations.json`):**  
```json
{
    "annotations": [
        {
            "question_id": 1,
            "image_id": "P01_01_00001.jpg",
            "answers": [
                {"answer": "scalpel", "confidence": "high", "answer_id": 1}
            ]
        }
    ]
}
```

---

## 📌 Key Features  
✅ **Multi-task annotations** (action, hands, tools, VQA)  
✅ **Structured folder hierarchy** for easy data loading  
✅ **Standard formats** (COCO for hands, YOLO for tools, VQA-JSON)  

---

## 📜 License  

CC BY-NC-SA 4.0