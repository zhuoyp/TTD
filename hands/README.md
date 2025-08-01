## How to Prepare a Video Dataset for MMDetection

This guide outlines the steps to process a video-based dataset and format it for object detection training with MMDetection.

-----

### 1\. Directory Setup

First, organize your dataset into the following directory structure. You may need to create these folders.

  * **Place raw video files** (e.g., `.mp4`) in split-specific folders:

      * `videos/train/`
      * `videos/val/`

  * **Place corresponding raw JSON annotation files** in subdirectories named after each video (without the file extension). This structure is required for the merging script.

      * `coco/train/video_name_1/annotation.json`
      * `coco/train/video_name_2/annotation.json`
      * `coco/val/video_name_3/annotation.json`

-----

### 2\. Extract Image Frames from Videos

Run the `create_dataset_images.py` script to convert your videos into image frames.

  * Before running, open the script and set the `split` variable (e.g., `split = 'train'`).
  * This script reads videos from `videos/{split}/` and saves the extracted frames into `coco/{split}/`.
  * Run the script for each of your data splits (e.g., `train` and `val`).

<!-- end list -->

```bash
python create_dataset_images.py
```

-----

### 3\. Create COCO Annotation Files

Next, run the `create_dataset_annotations.py` script to merge the individual JSON files into a single COCO-formatted annotation file for each split.

  * Ensure your annotation files are correctly placed as described in Step 1.
  * Open the script and set the `split` variable to match the folder you are processing.
  * The script will create a master annotation file (e.g., `train_annotations.json`) in your project's root directory.

<!-- end list -->

```bash
python create_dataset_annotations.py
```

After completing these steps, you will have your image frames in `coco/train/`, `coco/val/`, etc., and your master annotation files (`train_annotations.json`, `val_annotations.json`) ready for training.

-----

### 4\. Train a Model with MMDetection

Your dataset is now prepared for use with MMDetection.

1.  **Install MMDetection** by following the official instructions: [https://mmdetection.readthedocs.io/en/latest/get\_started.html](https://mmdetection.readthedocs.io/en/latest/get_started.html)

2.  **Update Config File** by modifying a model's config file to point to your dataset. Update the `data_root` and the `data` section to specify the paths to your images and annotation files.

3.  **Run Training Script** with your new config file to start training the model.

For more details on setting up configs and training scripts, see the MMDetection documentation: [https://mmdetection.readthedocs.io/en/latest/user\_guides/1\_config.html](https://www.google.com/search?q=https://mmdetection.readthedocs.io/en/latest/user_guides/1_config.html)