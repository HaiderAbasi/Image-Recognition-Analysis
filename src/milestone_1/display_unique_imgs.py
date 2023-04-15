import cv2
import os
import config
import pandas as pd
import ast

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
    
    num_duplicates = duplicate_counts.get(id, 0)
    
    # Load the image
    img_path = os.path.join(config.data_dir, id)
    img = cv2.imread(img_path)

    # Add the number of duplicates on the top right corner of the image
    text = f"Duplicates: {num_duplicates}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = img.shape[1] - text_size[0] - 20
    text_y = text_size[1] + 20
    rect_x = img.shape[1] - text_size[0] - 30
    rect_y = 10
    rect_w = text_size[0] + 20
    rect_h = text_size[1] + 20

    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), -1)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

    # Display the image
    cv2.imshow('Unique-img', img)
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