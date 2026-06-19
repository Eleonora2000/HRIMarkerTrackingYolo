# HRI Marker Tracking with YOLO

YOLO-based marker tracking algorithm developed for a Master's thesis at the Italian Institute of Technology (IIT) on physical Human–Robot Interaction.

## Overview

This project automatically tracks the motion of a human participant, a collaborative robot, and the manipulated object during collaborative experiments.

The algorithm processes image sequences using a trained YOLO model to detect three experimental markers:

- **H** – Human
- **R** – Robot
- **S** – Soap

For each frame, the pixel coordinates of the detected markers are extracted and exported for further analysis.

---

## Research Context

This software was developed as part of my Master's thesis in Mechatronic Engineering at Politecnico di Torino, carried out at the Italian Institute of Technology (IIT).

The project investigated **motor synchronization during physical human–robot collaboration**, comparing a collaborative industrial robot (UR5e) with a humanoid robot.

The trajectories extracted by this algorithm were used for the quantitative analysis of the collaborative task.

---

## Method

For every frame:

1. Detect the three experimental markers using a trained YOLO model.
2. Extract the pixel coordinates of each detected marker.
3. Save the trajectories into CSV files.
4. Optionally generate annotated images for visual inspection.

---

## Output

- Frame-by-frame marker coordinates (.csv)
- Optional annotated images showing YOLO detections

These trajectories were subsequently used to perform:

- motion trajectory reconstruction
- synchronization analysis
- range of motion (ROM) estimation
- statistical analysis of human–robot interaction

---

## Technologies

- Python
- YOLO
- OpenCV
- NumPy
