import cv2
import pandas as pd
import os
import config

# Read the CSV file
df = pd.read_csv(config.csv_display)


# Count the number of duplicates
num_duplicates = df['num_duplicates'].sum()

# Count the number of rows
num_rows = df.shape[0]

# Print the results
print(f"Number of duplicates: {num_duplicates}")
print(f"Number of unique Images: {num_rows}")
print(f"Total Images in image_duplicates{config.case_display}.csv : {num_duplicates + num_rows}")


# Iterate over the rows in the dataframe
for index, row in df.iterrows():
    # Load the image
    img_path = os.path.join(config.data_dir, row['image_id'])
    img = cv2.imread(img_path)

    # Add the number of duplicates on the top right corner of the image
    num_duplicates = row['num_duplicates']
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
