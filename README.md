# HRI Marker Tracking with YOLO

YOLO-based marker tracking script developed for human–robot interaction (HRI) experiments.

## Description

This script processes image sequences using a trained YOLO model to detect three specific markers:

- H (Human)
- S (Soap)
- R (Robot)

For each frame, the pixel coordinates of detected markers are extracted and saved to CSV files.

The resulting trajectories can be used for quantitative motion analysis and synchronization studies in collaborative tasks.

Optionally, the script can generate overlay images to visually inspect detection performance.

## Output

- CSV file containing frame-by-frame pixel coordinates
- Optional overlay images for detection validation
