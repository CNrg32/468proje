# BIL 468 Homework 1 - Artwork Object Detection with Template Matching

This project applies template matching for object detection on artwork images.

Dataset idea:
DEArt Dataset of European Art

Task:
Instead of painter or style classification, this project detects the location of a selected object class in artwork images using bounding box annotations.

Files:
- homework1_template_matching.py: Main Python code for dataset loading, preprocessing, template matching, validation, testing, metrics, and visualization.
- results/: Output folder for metrics and detected images.
- data/: Dataset folder.

Dataset structure:
data/
  raw/
    images/
      image files
    annotations.json

The annotation file is expected in COCO-style format with:
- images
- annotations
- categories

The selected object class should be changed in the TARGET_CLASS_NAME variable inside the Python file.