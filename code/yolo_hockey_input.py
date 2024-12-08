from ultralytics import YOLO
model = YOLO('./classes_12.pt')
results = model.predict('./penalty.mp4',save=True)
