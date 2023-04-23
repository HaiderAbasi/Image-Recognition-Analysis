import cv2
import os
import config
import pandas as pd
import ast

from src.utilities import put_Text,padded_resize,show_clr

from src.milestone_4.visual_sentiment_analysis import SentimentAnalyzer

s_analyzer = SentimentAnalyzer()

# Load the CSV file into a DataFrame
df = pd.read_csv(config.csv_display)

# Count the number of duplicates for each image
duplicate_counts = df['duplicate_of'].value_counts()

# Select rows where duplicate_of is '-'
unique_images = df.loc[df['duplicate_of'] == '-']

# Count the number of images with no duplicates
total_duplicates = duplicate_counts.sum() - duplicate_counts.get('-')

# Loop through each unique image and display its duplicate count
for index, row in unique_images.iterrows():
    id = row['image_id']
    
    dominant_clr =  ast.literal_eval(row['dominant_clr'])
    saturation = row['saturation']
    sat_level = row['sat_level']
    brightness = row['brightness']
    brightness_level = row['brightness_level']

    adults = row['Adults']
    children = row['Children']
    
    human_emotion = row['human_emotion']
    image_sentiment = row['image_sentiment']
    focus_subject = row['focus_subject']
    text_present = row['text_present']

    r,g,b= dominant_clr
    num_duplicates = duplicate_counts.get(id, 0)
    
    # Load the image
    img_path = os.path.join(config.data_dir, id)
    img = cv2.imread(img_path)

    # Add the number of duplicates on the top right corner of the image
    img_analysis_str = f"(Color,Sat,Brig): {dominant_clr, sat_level, brightness_level}\n{adults+children} People ({adults} adults & {children} children)\nHuman emotion: {human_emotion} - Image sentiment: {image_sentiment}\nFocus Subject: {focus_subject} - Text present: {text_present}"

    # Resize image to default size
    cv_img = padded_resize(img, (750,350))

    put_Text(cv_img,img_analysis_str,(10,10),bg_color=(0,0,0),outline_color=(255,255,255))
    
    clr = s_analyzer.map_to_dominant_color(dominant_clr)
    show_clr(cv_img,dominant_clr,clr)
    
    # Display the image
    cv2.imshow('Unique-img', cv_img)
    k = cv2.waitKey(0)
    if k==27:
        break

cv2.destroyAllWindows()

    
# Print the results
print("\n##### Stats #####")
print(f"Number of duplicates: {total_duplicates}")
print(f"Number of unique Images: {len(unique_images)}")
print(f"Total Images in image_duplicates{config.case_display}.csv : {total_duplicates + len(unique_images)}")
print("##### ----- #####\n")