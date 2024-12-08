# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:14:55 2024

@author: Raúl Vizcarra Chirinos
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from implement import TPS_ResNet_BiLSTM_Attn, Options
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.tesseract_recog import JerseyNumberExtractor

# MODEL INPUTS
model_path = '../last.pt'
video_path = '../video_input.mp4'
output_path = './output_video_trial4.mp4'
tracks_path = '../track_stubs.pkl'
classifier_path = '../hockey_team_classifier.pth'
metadata_output_path  = '../metadata.json'
pth_model = './hello.pth'

#***************** Loads models and ice rink coordinates**********************#
class_names = ['Referee', 'Tm_white', 'Tm_yellow',]

class HockeyAnalyzer:
    def __init__(self, model_path, classifier_path):
        self.model = YOLO(model_path)
        self.classifier = self.load_classifier(classifier_path)
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.rink_coordinates = np.array([[-450, 710], [2030, 710], [948, 61], [352, 61]])
        self.zone_white = [(180, 150), (1100, 150), (900, 61), (352, 61)]
        self.zone_yellow = [(-450, 710), (2030, 710), (1160, 150), (200, 150)]
        
        self.pixel_to_meter_conversion()
        self.previous_positions = {}
        self.team_stats = {
            'Tm_white': {'distance': 0, 'speed': [], 'count': 0, 'offensive_pressure': 0},
            'Tm_yellow': {'distance': 0, 'speed': [], 'count': 0, 'offensive_pressure': 0}
        }
        
#******************** Detect objects in each frame ***************************#
    def detect_frames(self, frames):
        batch_size = 20 
        detections = [] 
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
#*********************** Loads CNN Model**************************************#

    def load_classifier(self, classifier_path):
        model = CNNModel()
        model.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def predict_team(self, image):
        with torch.no_grad():
            output = self.classifier(image)
            _, predicted = torch.max(output, 1)
            predicted_index = predicted.item()
            team = class_names[predicted_index]
        return team


#*****************Pixel-based measurements to meters**************************#
    def pixel_to_meter_conversion(self):
        #Rink real dimensions in meters
        rink_width_m = 15
        rink_height_m = 30

        #Coordinates for rink dimensions
        left_pixel, right_pixel = self.rink_coordinates[0][0], self.rink_coordinates[1][0]
        top_pixel, bottom_pixel = self.rink_coordinates[2][1], self.rink_coordinates[0][1]

        #Conversion factors
        self.pixels_per_meter_x = (right_pixel - left_pixel) / rink_width_m
        self.pixels_per_meter_y = (bottom_pixel - top_pixel) / rink_height_m

        #Apply conversion factor
    def convert_pixels_to_meters(self, distance_pixels):
        return distance_pixels / self.pixels_per_meter_x, distance_pixels / self.pixels_per_meter_y
    
#**************** Speed: meters per second************************************#

    def calculate_speed(self, track_id, x_center, y_center, fps):
        current_position = (x_center, y_center)
        if track_id in self.previous_positions:
            prev_position = self.previous_positions[track_id]
            distance_pixels = np.linalg.norm(np.array(current_position) - np.array(prev_position))
            distance_meters_x, distance_meters_y = self.convert_pixels_to_meters(distance_pixels)
            speed_meters_per_second = (distance_meters_x**2 + distance_meters_y**2)**0.5 * fps
        else:
            speed_meters_per_second = 0
        self.previous_positions[track_id] = current_position
        return speed_meters_per_second

#*********** Locate player's position in Offensive zones**********************#

    def is_inside_zone(self, position, zone):
          x, y = position
          n = len(zone)
          inside = False
          p1x, p1y = zone[0]
          for i in range(n + 1):
              p2x, p2y = zone[i % n]
              if y > min(p1y, p2y):
                  if y <= max(p1y, p2y):
                      if x <= max(p1x, p2x):
                          if p1y != p2y:
                              xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                          if p1x == p2x or x <= xinters:
                              inside = not inside
              p1x, p1y = p2x, p2y
          return inside
      
        
#*******************************Performance metrics***************************#
    def draw_stats(self, frame):
         avg_speed_white = np.mean(self.team_stats['Tm_white']['speed']) if self.team_stats['Tm_white']['count'] > 0 else 0
         avg_speed_yellow = np.mean(self.team_stats['Tm_yellow']['speed']) if self.team_stats['Tm_yellow']['count'] > 0 else 0
         distance_white = self.team_stats['Tm_white']['distance']
         distance_yellow = self.team_stats['Tm_yellow']['distance']
    
         offensive_pressure_white = self.team_stats['Tm_white'].get('offensive_pressure', 0)
         offensive_pressure_yellow = self.team_stats['Tm_yellow'].get('offensive_pressure', 0)
         
         Pressure_ratio_W = offensive_pressure_white/distance_white   *100  if self.team_stats['Tm_white']['distance'] > 0 else 0
         Pressure_ratio_Y = offensive_pressure_yellow/distance_yellow *100  if self.team_stats['Tm_yellow']['distance'] > 0 else 0
    
         table = [
             ["", "Away_White", "Home_Yellow"],
             ["Average Speed\nPlayer", f"{avg_speed_white:.2f} m/s", f"{avg_speed_yellow:.2f} m/s"],
             ["Distance\nCovered", f"{distance_white:.2f} m", f"{distance_yellow:.2f} m"],
             ["Offensive\nPressure %", f"{Pressure_ratio_W:.2f} %", f"{Pressure_ratio_Y:.2f} %"],
         ]
    
         text_color = (0, 0, 0)  
         start_x, start_y = 10, 590 
         row_height = 30     # Manage Height between rows
         column_width = 150  # Manage Width  between rows
         font_scale = 1  
    
         def put_multiline_text(frame, text, position, font, font_scale, color, thickness, line_type, line_spacing=1.0):
             y0, dy = position[1], int(font_scale * 20 * line_spacing)  
             for i, line in enumerate(text.split('\n')):
                 y = y0 + i * dy
                 cv2.putText(frame, line, (position[0], y), font, font_scale, color, thickness, line_type)
        
         #Adjust multiline rows
         for i, row in enumerate(table):
             for j, text in enumerate(row):
                 if i in [1,2, 3]:  
                     put_multiline_text(
                         frame,
                         text,
                         (start_x + j * column_width, start_y + i * row_height),
                         cv2.FONT_HERSHEY_PLAIN,
                         font_scale,
                         text_color,
                         1,
                         cv2.LINE_AA,
                         line_spacing= 0.8  
                     )
                 else:
                     cv2.putText(
                         frame,
                         text,
                         (start_x + j * column_width, start_y + i * row_height),
                         cv2.FONT_HERSHEY_PLAIN,
                         font_scale,
                         text_color,
                         1,
                         cv2.LINE_AA,
                     )       
          
#********************* Track and update game stats****************************#

    def update_team_stats(self, team, speed, distance, position):
        if team in self.team_stats:
            self.team_stats[team]['speed'].append(speed)
            self.team_stats[team]['distance'] += distance
            self.team_stats[team]['count'] += 1

            if team == 'Tm_white':
                if self.is_inside_zone(position, self.zone_white):
                    self.team_stats[team]['offensive_pressure'] += distance
            elif team == 'Tm_yellow':
                if self.is_inside_zone(position, self.zone_yellow):
                    self.team_stats[team]['offensive_pressure'] += distance


#********* Ellipse for tracking players instead of Bounding boxes*************#
    def draw_ellipse(self, frame, bbox, color, track_id=None, team=None):
        y2 = int(bbox[3])
        x_center = (int(bbox[0]) + int(bbox[2])) // 2
        width = int(bbox[2]) - int(bbox[0])
    
        if team == 'Referee':
            color = (0, 255, 255)
            text_color = (0, 0, 0)
        else:
            color = (255, 0, 0)
            text_color = (255, 255, 255)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width) // 2, int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
    
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15
    
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
    
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            font_scale = 0.4
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness=2
            )

        return frame
    
#***************************Dashboard*****************************************#
    
    def draw_semi_transparent_rectangle(self, frame):
        overlay = frame.copy()
        alpha = 0.7  # Adjust Transparency
        
        bottom_left = (0, 710)
        bottom_right = (450, 710)
        upper_left = (0, 570)
        upper_right = (450, 570)
        
        border_color = (169, 169, 169)  # Color
        border_thickness = 3
        cv2.rectangle(frame, upper_left, bottom_right, border_color, border_thickness)
        cv2.rectangle(overlay, upper_left, bottom_right, (128, 128, 128), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        #Draw stats 
        self.draw_stats(frame)
 
#******************* Loads Tracked Data (pickle file )************************#

#     def analyze_video(self, video_path, output_path, tracks_path):
#           with open(tracks_path, 'rb') as f:
#               tracks = pickle.load(f)

#           cap = cv2.VideoCapture(video_path)
#           if not cap.isOpened():
#               print("Error: Could not open video.")
#               return
          
#           fps = cap.get(cv2.CAP_PROP_FPS)
#           frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#           frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#           # Codec and VideoWriter object
#           fourcc = cv2.VideoWriter_fourcc(*'XVID')
#           out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#           frame_num = 0
#           while cap.isOpened():
#               ret, frame = cap.read()
#               if not ret:
#                   break
              

#               mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#               cv2.fillConvexPoly(mask, self.rink_coordinates, 1)
#               mask = mask.astype(bool)
#               # Draw rink area
#               cv2.polylines(frame, [self.rink_coordinates], isClosed=True, color=(0, 255, 0), thickness=2)

#               #Get tracks from frame
#               player_dict = tracks["person"][frame_num]
#               for track_id, player in player_dict.items():
#                   bbox = player["bbox"]

#               #Check if players are within the ice rink
#                   x_center = int((bbox[0] + bbox[2]) / 2)
#                   y_center = int((bbox[1] + bbox[3]) / 2)

#                   if not mask[y_center, x_center]:
#                       continue  

# #***************************Team Prediction***********************************#
#                   x1, y1, x2, y2 = map(int, bbox)
#                   cropped_image = frame[y1:y2, x1:x2]
#                   cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#                   transformed_image = self.transform(cropped_pil_image).unsqueeze(0)
#                   team = self.predict_team(transformed_image)

# #************ Identify Teams, Players, assign labels**************************#
#                   self.draw_ellipse(frame, bbox, (0, 255, 0), track_id, team)
                  
#                   font_scale = 1  
#                   text_offset = -20  
                  
#                   #Identify Referee
#                   if team == 'Referee':
#                       rectangle_width = 60
#                       rectangle_height = 25
#                       x1_rect = x1
#                       x2_rect = x1 + rectangle_width
#                       y1_rect = y1 - 30
#                       y2_rect = y1 - 5
                  
#                       cv2.rectangle(frame,
#                                     (int(x1_rect), int(y1_rect)),
#                                     (int(x2_rect), int(y2_rect)),
#                                     (0, 0, 0),  
#                                     cv2.FILLED)
#                       text_color = (255, 255, 255)   
#                   else:
#                       if team == 'Tm_white':
#                           text_color = (255, 215, 0)  # White  Team: Blue labels
#                       else:
#                           text_color = (0, 255, 255)  # Yellow Team: Yellow labels
                  
#               #Draw Team labels
#                   cv2.putText(
#                       frame,
#                       team,
#                       (int(x1), int(y1) + text_offset), 
#                       cv2.FONT_HERSHEY_PLAIN,            
#                       font_scale,
#                       text_color,
#                       thickness=2
#                   )
                  
#                   speed = self.calculate_speed(track_id, x_center, y_center, fps)
                  
#               #Speed labels
#                   speed_font_scale = 0.8  
#                   speed_y_position = int(y1) + 20
#                   if speed_y_position > int(y1) - 5:
#                       speed_y_position = int(y1) - 5

#                   cv2.putText(
#                       frame,
#                       f"Speed: {speed:.2f} m/s",  
#                       (int(x1), speed_y_position),  
#                       cv2.FONT_HERSHEY_PLAIN,       
#                       speed_font_scale,
#                       text_color,
#                       thickness=2
#                   )

                
#                #Draw dashboard
#                   self.draw_semi_transparent_rectangle(frame)
                  
#                   distance = speed / fps
#                   position = (x_center, y_center)
#                   self.update_team_stats(team, speed, distance, position)
                    
#               #Process output video
#               out.write(frame)
#               frame_num += 1

#           cap.release()
#           out.release()

#********************CNN -Model - Architecture********************************#

    # def analyze_video(self, video_path, output_path, tracks_path, metadata_output_path):
    #     with open(tracks_path, 'rb') as f:
    #         tracks = pickle.load(f)

    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         print("Error: Could not open video.")
    #         return

    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #     # Codec and VideoWriter object
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    #     synthetic_metadata = {}  # To store metadata for all frames
    #     frame_num = 0

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    #         cv2.fillConvexPoly(mask, self.rink_coordinates, 1)
    #         mask = mask.astype(bool)

    #         # Draw rink area
    #         cv2.polylines(frame, [self.rink_coordinates], isClosed=True, color=(0, 255, 0), thickness=2)

    #         # Get tracks for the current frame
    #         player_dict = tracks["person"][frame_num]
    #         frame_metadata = {}

    #         for track_id, player in player_dict.items():
    #             bbox = player["bbox"]

    #             # Check if players are within the ice rink
    #             x_center = int((bbox[0] + bbox[2]) / 2)
    #             y_center = int((bbox[1] + bbox[3]) / 2)
    #             if not mask[y_center, x_center]:
    #                 continue

    #             # Predict the team
    #             x1, y1, x2, y2 = map(int, bbox)
    #             cropped_image = frame[y1:y2, x1:x2]
    #             cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #             transformed_image = self.transform(cropped_pil_image).unsqueeze(0)
    #             team = self.predict_team(transformed_image)

    #             # Get player name
    #             player_name = self.get_player_name(track_id)

    #             # Draw bounding ellipse and labels
    #             self.draw_ellipse(frame, bbox, (0, 255, 0), track_id, team)
    #             text_color = (255, 255, 255) if team == 'Referee' else (255, 215, 0) if team == 'Tm_white' else (0, 255, 255)

    #             # Draw team label
    #             cv2.putText(
    #                 frame,
    #                 team,
    #                 (x1, y1 - 20),
    #                 cv2.FONT_HERSHEY_PLAIN,
    #                 1,
    #                 text_color,
    #                 thickness=2
    #             )

    #             # Draw player name
    #             cv2.putText(
    #                 frame,
    #                 player_name,
    #                 (x1, y1 - 40),
    #                 cv2.FONT_HERSHEY_PLAIN,
    #                 1,
    #                 (255, 255, 255),
    #                 thickness=2
    #             )

    #             # Calculate speed
    #             speed = self.calculate_speed(track_id, x_center, y_center, fps)

    #             # Add speed label
    #             cv2.putText(
    #                 frame,
    #                 f"Speed: {speed:.2f} m/s",
    #                 (x1, y1 + 20),
    #                 cv2.FONT_HERSHEY_PLAIN,
    #                 0.8,
    #                 text_color,
    #                 thickness=2
    #             )

    #             # Update team stats
    #             distance = speed / fps
    #             position = (x_center, y_center)
    #             self.update_team_stats(team, speed, distance, position)

    #             # Save metadata
    #             frame_metadata[track_id] = {
    #                 "name": player_name,
    #                 "team": team,
    #                 "speed": round(speed, 2),
    #                 "distance": round(distance, 2)
    #             }

    #         # Draw the dashboard
    #         self.draw_semi_transparent_rectangle(frame)

    #         # Save metadata for the current frame
    #         synthetic_metadata[frame_num] = frame_metadata

    #         # Write processed frame to the output video
    #         out.write(frame)
    #         frame_num += 1

    #     # Save synthetic metadata to a JSON file

    #     with open(metadata_output_path, 'w') as meta_file:
    #         json.dump(synthetic_metadata, meta_file, indent=4)

    #     cap.release()
    #     out.release()
    #     print("Video analysis complete. Output saved to:", output_path)
    #     print("Metadata saved to:", metadata_output_path)

    # Supporting method for player names
    def analyze_video(self, video_path, output_path, tracks_path, metadata_output_path, pth_model):
        with open(tracks_path, 'rb') as f:
            tracks = pickle.load(f)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Codec and VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        synthetic_metadata = {}  # To store metadata for all frames
        frame_num = 0
        jersey_numbers = {}  # Store jersey numbers for each track_id
        while cap.isOpened():
            
            ret, frame = cap.read()
            if not ret:
                break
            print(frame_num)

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, self.rink_coordinates, 1)
            mask = mask.astype(bool)

            # Draw rink area
            cv2.polylines(frame, [self.rink_coordinates], isClosed=True, color=(0, 255, 0), thickness=2)

            # Get tracks for the current frame
            player_dict = tracks["person"][frame_num]
            frame_metadata = {}

            for track_id, player in player_dict.items():
                bbox = player["bbox"]

                # Check if players are within the ice rink
                x_center = int((bbox[0] + bbox[2]) / 2)
                y_center = int((bbox[1] + bbox[3]) / 2)
                if not mask[y_center, x_center]:
                    continue

                # Initialize JerseyNumberExtractor only once
                extractor = JerseyNumberExtractor()

                # Extract the jersey number (if it hasn't been detected yet for this track_id)
                if track_id not in jersey_numbers:
                    try:
                        detected_number = extractor.get_jersey_number(frame, bbox, pth_model)
                        if detected_number:  # Store only if a number was detected
                            jersey_numbers[track_id] = detected_number
                    except ValueError as e:
                        print(f"Error processing track_id {track_id}: {e}")
                        detected_number = None
                else:
                    detected_number = jersey_numbers[track_id]

                # Predict the team
                x1, y1, x2, y2 = map(int, bbox)
                cropped_image = frame[y1:y2, x1:x2]
                cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                transformed_image = self.transform(cropped_pil_image).unsqueeze(0)
                team = self.predict_team(transformed_image)

                # Draw bounding ellipse and labels
                self.draw_ellipse(frame, bbox, (0, 255, 0), track_id, team)
                text_color = (255, 255, 255) if team == 'Referee' else (255, 215, 0) if team == 'Tm_white' else (0, 255, 255)

                # Draw team label and jersey number
                if detected_number:
                    cv2.putText(frame, f"#{detected_number}", (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=2)
                cv2.putText(frame, team, (x1, y1 - 40), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=2)

                # Calculate speed
                speed = self.calculate_speed(track_id, x_center, y_center, fps)

                # Add speed label
                cv2.putText(
                    frame,
                    f"Speed: {speed:.2f} m/s",
                    (x1, y1 + 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.8,
                    text_color,
                    thickness=2
                )

                # Update team stats
                distance = speed / fps
                position = (x_center, y_center)
                self.update_team_stats(team, speed, distance, position)

                # Save metadata
                frame_metadata[track_id] = {
                    "jersey_number": detected_number,
                    "team": team,
                    "bbox": bbox,
                    "speed": round(speed, 2),
                    "distance": round(distance, 2)
                }

            # Draw the dashboard
            self.draw_semi_transparent_rectangle(frame)

            # Save metadata for the current frame
            synthetic_metadata[frame_num] = frame_metadata

            # Write processed frame to the output video
            out.write(frame)
            frame_num += 1

        # Save synthetic metadata to a JSON file
        with open(metadata_output_path, 'w') as meta_file:
            json.dump(synthetic_metadata, meta_file, indent=4)

        cap.release()
        out.release()
        print("Video analysis complete. Output saved to:", output_path)
        print("Metadata saved to:", metadata_output_path)
    def get_player_name(self, track_id):
        """
        Returns the player's name for a given track ID.
        Replace with your logic for fetching names (e.g., from a database or mapping).
        """
        player_names = {
            1: "Alice",
            2: "Bob",
            3: "Charlie",
            4: "Diana",
            # Add more mappings as needed
        }
        return player_names.get(track_id, "Unknown Player")
    def prepare_image(self, frame, bbox):
    # Crop and resize the region of interest
        roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        resized_image = cv2.resize(roi, (150, 150))

        # Convert to grayscale if required
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Normalize and convert to tensor
        tensor_image = torch.tensor(gray_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        return tensor_image
    def get_jersey_number(self, frame, bbox, pth_model):
    # """
    # Extracts and recognizes the jersey number from a given bounding box in the frame.
    
    # Args:
    #     frame (ndarray): The video frame.
    #     bbox (tuple): The bounding box coordinates (x1, y1, x2, y2).
    #     pth_model (torch.nn.Module): The pre-trained model for jersey number recognition.

    # Returns:
    #     str: The detected jersey number.
    # """
    # Extract the region of interest (ROI)
        x1, y1, x2, y2 = map(int, bbox)
        cropped_image = frame[y1:y2, x1:x2]

        # Convert the image to PIL format and transform it for the model
        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        transformed_image = self.prepare_image(frame, bbox)
    
    # Convert RGB to Grayscale
        grayscale_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        transformed_image = grayscale_transform(transformed_image)

        # Ensure input has correct dimensions
        transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension


        opt = Options()
# Instantiate the model
        pth_model = TPS_ResNet_BiLSTM_Attn(opt)

        # Load the state dictionary
        state_dict = torch.load("./hello.pth")
        pth_model.load_state_dict(state_dict, strict=False)

        # Switch to evaluation mode
        pth_model.eval()
        # with torch.no_grad():
        #     prediction = pth_model(transformed_image, dummy_text, is_train=False)
        #     jersey_number = prediction.argmax(dim=1).item()

        # return str(jersey_number)
        with torch.no_grad():
            visual_feature = pth_model.FeatureExtraction(transformed_image)
            contextual_feature = visual_feature.squeeze(3)  # Adjust shape as needed
            jersey_number = torch.argmax(contextual_feature, dim=1)  # Simple classification

        return jersey_number


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(class_names))  
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#*********Execute YOLO-HockeyAnalyzer/classifier and Save Output**************#
print(model_path, classifier_path, video_path, output_path, tracks_path, metadata_output_path, pth_model)
analyzer = HockeyAnalyzer(model_path, classifier_path)
analyzer.analyze_video(video_path, output_path, tracks_path, metadata_output_path, pth_model)

