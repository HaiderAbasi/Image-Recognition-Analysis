from deepface import DeepFace
import cv2
import config

def get_general_emotion(faces_data):
    emotions = [face_data['dominant_emotion'] for face_data in faces_data]
    emotion_count = {emotion:emotions.count(emotion) for emotion in emotions}
    sorted_emotions = sorted(emotion_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_emotions[0][0]

def analyze_faces(image):
    
    general_emotion = "-"

    try:
        faces_data = DeepFace.analyze(img_path = image, 
            actions = ['emotion'],
            enforce_detection= True,
            detector_backend="opencv",
            silent= True)
    except Exception as e:
        if config.debug:
            print("Exception occured during face analysis = ",e)
        return general_emotion
    
    if config.display_face_analysis:
        # Define the colors for the face rectangle and emotion label
        green = (0, 255, 0)
        red = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # If its an image path read image from disk
        if isinstance(image,str):
            image = cv2.imread(image)
            
        # Loop through the list of faces detected in the image
        for face_data in faces_data:

            # Extract the coordinates and emotions for the current face_data
            x, y, w, h = face_data['region']['x'], face_data['region']['y'], face_data['region']['w'], face_data['region']['h']
            emotions = face_data['emotion']

            # Draw a rectangle around the detected face_data
            cv2.rectangle(image, (x, y), (x + w, y + h), green, thickness)

            # Add a text label indicating the dominant emotion detected
            dominant_emotion = face_data['dominant_emotion']
            label = f'{dominant_emotion}: {emotions[dominant_emotion]:.2f}%'
            cv2.putText(image, label, (x, y - 10), font, font_scale, red, thickness, cv2.LINE_AA)

        # Display the annotated image
        cv2.imshow('Annotated Image', image)
        cv2.waitKey(0)
        
    if len(faces_data)>0:
        general_emotion = get_general_emotion(faces_data)
     
    return general_emotion
    
    
    
if __name__ == "__main__":
    
    #img = cv2.imread(r"src\milestone_4\happyandangry.png")
    img = cv2.imread(r"src\milestone_4\human-emotion-facts.jpg")
    
    general_emotion = analyze_faces(img)
    
    print("general_emotion =  " ,general_emotion)