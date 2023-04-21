import pandas as pd
import numpy as np
import concurrent.futures
import time
import os
import cv2
import config
import gdown

from PIL import Image

from milestone_1.img_duplicate_checker_fast import identify_duplicates_nd_updates_csv
from milestone_2.color_analysis import perform_color_analysis
from milestone_3.people_counter import PeopleCounter,args
from milestone_3.age_classification import AgeClassifier
from milestone_4.face_analysis import analyze_faces
from milestone_4.visual_sentiment_analysis import SentimentAnalyzer
from milestone_4.deep_sentiment_analysis import compute_img_sentiment

from utilities import put_Text,padded_resize,show_clr


class ImageAnalysis:
    
    def __init__(self):    
        self.pc = PeopleCounter()
        self.img_sentiment_analyzer = SentimentAnalyzer()
        self.img = None
        
    emotion_to_sentiment = {"angry": "negative","disgust": "negative","fear": "negative",
                            "happy": "positive","sad": "negative","surprise": "positive",
                            "neutral": "neutral"}
           
    # Define the function to process each image
    def analyze(self,row):
        image_id = row['image_id']
        # Load the image
        img_path = os.path.join(config.data_dir, image_id)
        self.img = Image.open(img_path)

        # Milestone 2: Perform image color Analysis
        dominant_clr, saturation,sat_level,brightness, brightness_level = perform_color_analysis(self.img,img_path)

        # Milestone 3: Perform people count
        people_boxes,focus_subject = self.pc.count(img_path)
        
        # Milestone 4: Perform sentiment analysis
        human_emotion = "-"
        if len(people_boxes) > 0:
            human_emotion = analyze_faces(img_path)
        
        _,clr = self.img_sentiment_analyzer.analyze(brightness,saturation,human_emotion,dominant_clr)
        
        if human_emotion =="-":
            image_sentiment = compute_img_sentiment(img_path,config.deep_sentiment_analyzer)
        else:
            focus_subject = "-"
            image_sentiment = self.emotion_to_sentiment[human_emotion]


        # Return the result as a tuple
        return (image_id, dominant_clr, clr,saturation,sat_level, brightness,brightness_level,focus_subject,people_boxes,human_emotion,image_sentiment)


def download_missing_model_files():
    """Download missing model files from Google Drive."""
    models_dir = os.path.join(os.getcwd(), 'data', 'models')
    model_files = ['yolox_s.pth', 'yolox_nano.pth', 'mobilenet_v2_adult_child_classifier.h5','hybrid_finetuned_fc6+.pth']  # replace with actual file names
    files_id = ['1N9Ymq95gPszQ94mAxgEUPc3Z9R5P-_nM', '1rcOHMnxXHVQ3fhMpGtsvalXY6GMu1SEA', '1VxfBUWe2VJgvNps-t3ccwoM7ccSMJzMW','1iHttZ7jVjHkNQyJO8X1nPq5yNd2emHoQ']  # replace with actual file names
    
    for i, file in enumerate(model_files):
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            print(f'{file} not found. Downloading from Google Drive...')
            file_id = files_id[i]  # replace with the actual file ID from the Google Drive link
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, file_path, quiet=False)
            print(f'{file} downloaded successfully!')


def main():
    # Download required model files
    download_missing_model_files()
    
    #args.nms =0.45
    args.conf = 0.5

    image_analysis = ImageAnalysis()
    age_classifier = AgeClassifier()
    

    if not os.path.isfile(config.csv_ptest):
        # Task 1: Create csv by writing all images ids with their respective duplicate on second column
        identify_duplicates_nd_updates_csv()
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(config.csv_ptest)

    # Select rows where duplicate_of is '-'
    unique_images = df.loc[df['duplicate_of'] == '-']

    # Create a ProcessPoolExecutor with the specified number of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(image_analysis.analyze, row) for _, row in unique_images.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            image_id, dominant_clr,clr, sat,sat_level, brightness, brightness_level,focus_subject,people_boxes,human_emotion,image_sentiment = future.result()
            
            # Load the image
            img_path = os.path.join(config.data_dir, image_id)
            img = Image.open(img_path)
            adults = 0
            children = 0
            for x,y,x2,y2 in people_boxes:
                person = img.crop((x, y, x2, y2))
                class_id = age_classifier.predict(person)
                if class_id ==0:
                    adults +=1
                else:
                    children +=1

            # Add a new column for the dominant color
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'dominant_clr'] = str(dominant_clr)

            # Add a new column for the saturation
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'saturation'] = sat

            # Add a new column for the max saturation
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'sat_level'] = sat_level

            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'brightness'] = brightness

            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'brightness_level'] = brightness_level

            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'Adults'] = adults

            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'Children'] = children
            
            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'human_emotion'] = human_emotion

            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'image_sentiment'] = image_sentiment
            
            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'focus_subject'] = focus_subject



            if config.display:
                # Load the image
                img_path = os.path.join(config.data_dir, image_id)
                img = Image.open(img_path)
                # Convert PIL image to OpenCV format
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Add the number of duplicates on the top right corner of the image
                img_analysis_str = f"(Color,Sat,Brig): {dominant_clr, sat_level, brightness_level}\n{adults+children} People ({adults} adults & {children} children)\nHuman emotion: {human_emotion} - Image sentiment: {image_sentiment}\nFocus Subject: {focus_subject}"

                # Resize image to default size
                cv_img = padded_resize(cv_img, (750,350))

                put_Text(cv_img,img_analysis_str,(10,10),bg_color=(0,0,0),outline_color=(255,255,255))

                show_clr(cv_img,dominant_clr,clr)
                
                # Display the image
                cv2.imshow('Unique-img', cv_img)
                k = cv2.waitKey(0)
                if k==27:
                    break
        

    if not config.display:
        # Write the dataframe to a CSV file
        df['saturation'] = df['saturation'].astype(int)
        df['brightness'] = df['brightness'].astype(int)
        df['Adults'] = df['Adults'].astype(int)
        df['Children'] = df['Children'].astype(int)
        df.to_csv(config.csv_ptest, index=False)
    else:
        cv2.destroyAllWindows()


        


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Time it took to perform sentiment-analysis is {time.time()-start} secs")