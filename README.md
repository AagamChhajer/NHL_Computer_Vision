# NHL_Assignment_Submission

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
4. [Architecture](#architecture)
5. [Tasks](#tasks)
    - [Task 1: Player Detection](#task-1-player-detection)
    - [Task 2: Puck Tracking](#task-2-puck-tracking)
    - [Task 3: Player Speed Estimation](#task-3-player-speed-estimation)
    - [Task 4: Zone Occupancy Analysis](#task-4-zone-occupancy-analysis)
    - [Task 5: Goal Detection](#task-5-goal-detection)
    - [Task 6: Penalty Event Detection](#task-6-penalty-event-detection)
6. [Advancements](#advancements)
7. [Shortcomings](#shortcomings)
8. [Contributions](#contributions)
9. [Acknowledgements](#acknowledgements)

---

## Introduction
**Brief overview of the project.**
- **Purpose:** The project aims to develop a set of computer vision and machine learning tasks to analyze and enhance the experience of watching hockey games.
- **Problem it solves:** Automates key processes such as player detection, puck tracking, and event recognition, saving time for analysts and coaches.
- **High-level benefits:** Provides actionable insights, highlights key moments, and improves the understanding of game dynamics in real-time.

![Project Overview](path/to/overview_image.png)

---

## Features
- **Player Detection:** Detect and classify players on the ice.
- **Puck Tracking:** Track the puck’s movement and predict future positions.
- **Player Speed Estimation:** Calculate player speed based on their movement.
- **Zone Occupancy Analysis:** Analyze the time each team spends in different zones.
- **Goal Detection:** Automatically detect when a goal is scored during gameplay.
- **Penalty Event Detection:** Identify and categorize penalty events during the game.

---

## Getting Started

### Prerequisites
List any prerequisites for setting up the project:
- Python 3.8+
- OpenCV 4.x
- PyTorch 1.10+
- YOLOv8 Pre-trained Model

## Tasks

### Task 1: Player Detection  
**Objective:**  
Detect and classify players on the ice in real-time.  

**Approach:**  
1. **Preprocessing:** Extracted frames from video footage and converted them to grayscale for computational efficiency.  
2. **Model:** Leveraged YOLOv8 for real-time object detection, trained on annotated hockey game datasets.  
3. **Output:** Bounding boxes with player labels (`Team A`, `Team B`, `Referee`).  

![Player Detection](path/to/player_detection_image.png)  

**Video Demonstration:**  
[![Player Detection Video](path/to/video_thumbnail.png)](https://youtu.be/player_detection_example)

---

### Task 2: Puck Tracking  
**Objective:**  
Track the movement of the puck throughout the game.  

**Approach:**  
1. **Segmentation:** Used HSV color space to isolate the puck based on its color.  
2. **Model:** Implemented Kalman Filters to predict the puck’s position during occlusions.  
3. **Output:** Real-time trajectory of the puck overlaid on the game footage.  

![Puck Tracking](path/to/puck_tracking_image.png)  

**Video Demonstration:**  
[![Puck Tracking Video](path/to/video_thumbnail.png)](https://youtu.be/puck_tracking_example)

---

### Task 3: Player Speed Estimation  
**Objective:**  
Calculate the speed of players during the game.  

**Approach:**  
1. **Coordinate Extraction:** Used the bounding box center coordinates from YOLOv8 outputs.  
2. **Conversion:** Converted pixel-based movements to real-world distances using camera calibration techniques.  
3. **Output:** Speed in km/h displayed near each player.  

![Speed Estimation](path/to/speed_estimation_image.png)  

**Video Demonstration:**  
[![Speed Estimation Video](path/to/video_thumbnail.png)](https://youtu.be/speed_estimation_example)

---

### Task 4: Zone Occupancy Analysis  
**Objective:**  
Analyze how much time each team spends in different zones of the rink.  

**Approach:**  
1. **Zone Division:** Divided the rink into three zones (defensive, neutral, offensive).  
2. **Tracking:** Mapped player positions to zones using pre-defined coordinates.  
3. **Output:** Heatmaps showing occupancy percentage by team.  

![Zone Occupancy](path/to/zone_occupancy_image.png)  

**Video Demonstration:**  
[![Zone Occupancy Video](path/to/video_thumbnail.png)](https://youtu.be/zone_occupancy_example)

---

### Task 5: Goal Detection  
**Objective:**  
Detect when a goal is scored during gameplay.  

**Approach:**  
1. **Event Detection:** Monitored the puck’s position relative to goal coordinates.  
2. **Audio Analysis:** Cross-validated with crowd noise spikes using an audio classifier.  
3. **Output:** Timestamped goal events with video highlights.  

![Goal Detection](path/to/goal_detection_image.png)  

**Video Demonstration:**  
[![Goal Detection Video](path/to/video_thumbnail.png)](https://youtu.be/goal_detection_example)

---

### Task 6: Penalty Event Detection  
**Objective:**  
Identify and categorize penalty events during the game.  

**Approach:**  
1. **Player Behavior Analysis:** Used pose estimation to detect specific gestures indicating penalties (e.g., high stick).  
2. **Referee Signal Recognition:** Detected referee’s gestures and validated with game logs.  
3. **Output:** Highlighted video clips of penalty events with descriptive labels.  

![Penalty Detection](path/to/penalty_detection_image.png)  

**Video Demonstration:**  
[![Penalty Detection Video](path/to/video_thumbnail.png)](https://youtu.be/penalty_detection_example)

### Installation
Step-by-step guide for installing the project locally:
```bash
git clone https://github.com/username/project-name.git
cd project-name
pip install -r requirements.txt

