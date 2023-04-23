import easyocr

class Text_detector:
    def __init__(self,lang_list = ['en']):
        # initialize the EasyOCR reader to detect all languages
        self.reader = easyocr.Reader(lang_list,recognizer=False)


    def is_text_present(self,image):
        
        availability = False
        
        # Detect text locations in the image
        locations = self.reader.detect(image)
        bboxes = locations[0][0]
        if bboxes:
            availability = True
        
        return availability