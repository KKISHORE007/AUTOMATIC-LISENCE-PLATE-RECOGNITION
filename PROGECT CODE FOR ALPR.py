import cv2
import pytesseract
import re

class LicensePlateRecognizer:
    def __init__(self):
        # Configure Tesseract path (change as needed)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize plate detector (Haar cascade or DNN)
        self.plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
        # Alternative: self.net = cv2.dnn.readNet('yolo-license_plate.cfg', 'yolo-license_plate.weights')

    def detect_plates(self, img):
        """Detect license plates in image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 5)
        # Alternative DNN approach:
        # blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True)
        # self.net.setInput(blob)
        # plates = self.net.forward()
        return plates

    def preprocess_plate(self, plate_img):
        """Enhance plate image for OCR"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh

    def recognize_plate(self, plate_img):
        """Perform OCR on license plate"""
        processed = self.preprocess_plate(plate_img)
        text = pytesseract.image_to_string(processed, 
                                         config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        # Clean and validate text
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        return text if len(text) > 4 else None

    def process_image(self, img_path):
        """Complete ALPR pipeline"""
        img = cv2.imread(img_path)
        plates = self.detect_plates(img)
        
        results = []
        for (x,y,w,h) in plates:
            plate_img = img[y:y+h, x:x+w]
            text = self.recognize_plate(plate_img)
            if text:
                results.append({'text': text, 'coordinates': (x,y,w,h)})
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow('ALPR Results', img)
        cv2.waitKey(0)
        return results

if __name__ == "__main__":
    alpr = LicensePlateRecognizer()
    result = alpr.process_image('car.jpg')
    print("Detected plates:", result)