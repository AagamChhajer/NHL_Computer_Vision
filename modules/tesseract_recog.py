import cv2
import pytesseract

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

class JerseyNumberExtractor:
    def __init__(self):
        pass

    def preprocess_image(self, frame, bbox):
        """Preprocess the cropped player bounding box for Tesseract OCR."""
        x1, y1, x2, y2 = map(int, bbox)

        # Crop the bounding box region
        cropped = frame[y1:y2, x1:x2]

        # Convert to grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to enhance the digits
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        return thresh

    def get_jersey_number(self, frame, bbox, pth_model=None):
        """
        Extract jersey number from the player's bounding box using Tesseract OCR.

        Parameters:
        - frame: The full video frame (numpy array).
        - bbox: The bounding box coordinates (x1, y1, x2, y2).
        - pth_model: Placeholder for PyTorch model, not used here.

        Returns:
        - jersey_number: Detected jersey number as a string.
        """
        # Preprocess the cropped image
        preprocessed_image = self.preprocess_image(frame, bbox)

        # Use Tesseract to perform OCR on the preprocessed image
        config = "--psm 6 -c tessedit_char_whitelist=0123456789"  # Only recognize digits
        jersey_number = pytesseract.image_to_string(preprocessed_image, config=config).strip()

        return jersey_number