import cv2
import pytesseract

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

class JerseyNumberExtractor:
    def __init__(self):
        pass

    def preprocess_image(self, frame, bbox):
    
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        return thresh

    def get_jersey_number(self, frame, bbox, pth_model=None):

        preprocessed_image = self.preprocess_image(frame, bbox)

        config = "--psm 6 -c tessedit_char_whitelist=0123456789" 
        jersey_number = pytesseract.image_to_string(preprocessed_image, config=config).strip()

        return jersey_number