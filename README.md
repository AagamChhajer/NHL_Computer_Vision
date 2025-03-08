# NHL_Assignment_Submission 🏒  

AI-powered computer vision system for hockey analysis. Automates player detection, puck tracking, and event recognition to enhance game understanding in real time.

----------------------------------------------

## Features
- **Player Detection:** Detect and classify players on the ice.
- **Player Speed Estimation:** Calculate player speed based on their movement.
- **Detail Analysis:** Analyze the offensive pressure and distance covered for each  team.
- **Goal Detection:** Automatically detect when a goal is scored during gameplay.
- **Object Detection:** Detects objects such as goalpost, hockey_stick and puck.

[![Output](/assets/player_name.jpg)](/output/output_video_trial3.mp4)
---
# Model Architectures

## TPS_ResNet_BiLSTM_Attn [Model Implementation](code/ocr_implement.ipynb)
![](/assets/ocr_parameters.png)


  ### Transformation
  #### TPS_SpatialTransformerNetwork  
  - **LocalizationNetwork**:
    - **Conv Layers**:
      - Conv2d (1 → 64) → BatchNorm2d → ReLU → MaxPool2d
      - Conv2d (64 → 128) → BatchNorm2d → ReLU → MaxPool2d
      - Conv2d (128 → 256) → BatchNorm2d → ReLU → MaxPool2d
      - Conv2d (256 → 512) → BatchNorm2d → ReLU → AdaptiveAvgPool2d
    - **Fully Connected Layers**:
      - Linear (512 → 256) → ReLU
      - Linear (256 → 40)
  - **GridGenerator**

  ### FeatureExtraction
  #### ResNet_FeatureExtractor
  - **Conv Layers**:
    - Conv2d (1 → 32) → BatchNorm2d → ReLU
    - Conv2d (32 → 64) → BatchNorm2d → ReLU → MaxPool2d
  - **ResNet Layers**:
    - **Layer 1**:
      - BasicBlock (64 → 128)
    - **Layer 2**:
      - BasicBlock (128 → 256)
    - **Layer 3**:
      - BasicBlock (256 → 512)
    - **Layer 4**:
      - BasicBlock (512 → 512)

  ### Sequence Modeling
  - BiLSTM (Input → Hidden)

  ### Prediction
  - Fully Connected Layer

#
## CNN Classifier Model Overview [Model Implementation](code/team_classifier.py)

  The CNN architecture consists of three convolutional layers followed by pooling, activation functions, and fully connected layers. It is designed for classifying images into three classes: `team_referee`, `team_away`, and `team_home`.

  ### 1. **Convolutional Layers**
  - **Conv1**: 
    - Input Channels: 3 (RGB image)
    - Output Channels: 32
    - Kernel Size: 3x3
    - Padding: 1
  - **Conv2**:
    - Input Channels: 32
    - Output Channels: 64
    - Kernel Size: 3x3
    - Padding: 1
  - **Conv3**:
    - Input Channels: 64
    - Output Channels: 128
    - Kernel Size: 3x3
    - Padding: 1

  ### 2. **Pooling Layers**
  - **Max Pooling**:
    - Kernel Size: 2x2
    - Stride: 2
    - Applied after each convolutional layer.

  ### 3. **Fully Connected Layers**
  - **Fully Connected 1**:
    - Input Features: (128 x 18 x 128) (from the final pooling layer)
    - Output Features: 512
  - **Dropout**:
    - Probability: 0.5 (to reduce overfitting)
  - **Output Layer**:
    - Input Features: 512
    - Output Features: 3 (for three classes)

  ### Activation Function
  - **ReLU (Rectified Linear Unit)**:
    - Applied after each convolutional layer and the first fully connected layer.

  ### Forward Propagation Flow
  1. Input Image 
  2. **Conv1 → ReLU → Max Pooling**
  3. **Conv2 → ReLU → Max Pooling**
  4. **Conv3 → ReLU → Max Pooling**
  5. Flatten the tensor for the fully connected layers.
  6. **Fully Connected 1 → ReLU → Dropout**:
    - Output: 512
  7. **Output Layer**:
    - Output: 3 (class probabilities).

---

## Complete Code Representation
-  ```python
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 18 * 18, 512)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 3)  # Three classes
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 18 * 18)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x 

---
## Tasks

### Task 1: Player Detection  
**Objective:**  
Detect and classify players on the ice in real-time.  

**Approach:**  
1. **Preprocessing:** Extracted frames from video footage and converted them to grayscale for computational efficiency.  
2. **Model:** Leveraged YOLO11 for real-time object detection, trained on annotated hockey game datasets.  
3. **Output:** Bounding boxes with player labels (`Team White`, `Team Yellow`, `Referee`). 


**Video Demonstration:**  
[![Player Detection Video](/assets/player_detection.png)](/output/player_detection.mp4)
[![Player Detection Video 2](/assets/player_detection_2.png)](/output/player_detection_2.mp4)
---

### Task 2: Puck and Hockey Stick Detection   
**Objective:**  
Detected hockey stick and puck throughout the game.  

**Puck Video Demonstration:**  
[![Puck Detection Video](/assets/puck_1.png)](/output/ball.mp4)
[![Puck Detection Video 2](/assets/puck_2.png)](/output/puck_2.mp4)

**Hockey Stick Video Demonstration:**  
[![Stick Detection Video](/assets/stick_detection.png)](/output/hockeystick_detection.mp4)

---

### Task 3: Goal Detection  
**Objective:**  
Detect when a goal is scored during gameplay.  

**Approach:**  
- **Event Detection:** Monitored the puck’s position relative to goal coordinates.  


![Goal Detection](/assets/goal_approach.png)(/code/goal_tracker.ipynb)  

**Video Demonstration:**  
[![Goal Detection Video](/assets/goal_output.png)](/output/goal_output.mp4)


---

## Datasets Used

### 1. Home, Away, and Referee Dataset
- **Description:** This dataset contains frames of players and referees, in separate folders labeled as home team, away team, and referees. It helps in player detection and classification of team affiliations.
- **Features:** 
  - Annotated frames with labels: `Home`, `Away`, `Referee`
- **Usage in Project:** 
  - This dataset is used to train the CNN model for detecting players on the ice and distinguishing between players from different teams and referees.
  - It provides team-specific information, which is crucial for tasks such as **Player Detection** and **Team Analysis**.
- **Training Dataset**
  - Referee
![Refree](/assets/referee.png)
  - Away Team
![Away Team](/assets/away_team.png)
  - Home Team
![Home Team](/assets/home_team.png)
- **Validation Dataset**
![](/assets/validation.png)

### 2. Jersey Number Dataset
- **Description:** This dataset contains images of hockey players with labeled jersey numbers, enabling the identification and classification of individual players based on their jerseys.
- **Features:** 
  - Labeled frames showing players with clear visibility of their jersey numbers.
  - Txt Files with corresponding coordinates.
- **Usage in Project:**
  - Used to track individual players and identify them based on their unique jersey numbers.
  - Helps in tasks such as **Player Detection** and **Player Speed Estimation**, enabling differentiation between players within the same team and providing personalized data for performance metrics.
- **Dataset Sample**
  - 3_925_1
![Image](/assets/ocr_dataset.png)

### 3. Roboflow Datasets
- Combined Dataset (Self Merged)
  - https://universe.roboflow.com/a-tgxjv/nhl-1996-3k7nm/dataset/3
![Class](/assets/class_distribution.png)
![Split](/assets/dataset_processing.png)
- Hockey Stick Dataset (Available on Roboflow Universe)
  - https://universe.roboflow.com/hokey-stick-recognition/our-da/dataset/4
![Class](/assets/stick_dataset.png)
![Split](/assets/stick_split.png)
---

## Future Enhancements 
- Use Supervision Library along with LSTM to track if the puck crosses the boundary and repetetive frames.
- Also Implement audio analysis which can use audience noise to predict if any event has occured.
- Can extract data from a steady cam with the whole view of the field

## Shortcomings
- Availability of quality datasets for free usage on Internet 
- Resource Bottlenecks
- A constant POV of the game (the current camera angle and pov are too unstable for the analysis)



## Acknowledgements
- Player Tracking and Identification in Ice-Hockey (https://arxiv.org/pdf/2110.03090)
- FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf)
- FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf)
- FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)
- https://github.com/clovaai/deep-text-recognition-benchmark
