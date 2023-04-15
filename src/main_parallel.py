import pandas as pd
import numpy as np
import concurrent.futures
import time
import os
import cv2
import config

from PIL import Image

from milestone_1.img_duplicate_checker_fast import identify_duplicates_nd_updates_csv
from milestone_2.color_analysis import perform_color_analysis


# Define the function to process each image
def analyze_img(row):
    image_id = row['image_id']
    # Load the image
    img_path = os.path.join(config.data_dir, image_id)
    img = Image.open(img_path)

    # Milestone 2: Perform image color Analysis
    dominant_clr, sat_level, brightness_level = perform_color_analysis(img,img_path)
    
    # Return the result as a tuple
    return (image_id, dominant_clr, sat_level, brightness_level)


def main():
    # Task 1: Create csv by writing all images ids with their respective duplicate on second column
    identify_duplicates_nd_updates_csv()
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(config.csv_ptest)

    # Select rows where duplicate_of is '-'
    unique_images = df.loc[df['duplicate_of'] == '-']

    display = False
    # Create a ProcessPoolExecutor with the specified number of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(analyze_img, row) for _, row in unique_images.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            image_id, dominant_clr, sat_level, brightness_level = future.result()
            
            # Add a new column for the dominant color
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'dominant_clr'] = str(dominant_clr)

            # Add a new column for the max saturation
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'sat_level'] = sat_level

            # Add a new column for the max brightness
            df.loc[(df['image_id'] == image_id) | (df['duplicate_of'] == image_id), 'brightness_level'] = brightness_level
        
            if display:
                # Load the image
                img_path = os.path.join(config.data_dir, image_id)
                img = Image.open(img_path)
                # Convert PIL image to OpenCV format
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Add the number of duplicates on the top right corner of the image
                text = f"(Color,Sat,Brig): {dominant_clr, sat_level, brightness_level}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = cv_img.shape[1] - text_size[0] - 20
                text_y = text_size[1] + 20
                rect_x = cv_img.shape[1] - text_size[0] - 30
                rect_y = 10
                rect_w = text_size[0] + 20
                rect_h = text_size[1] + 20
                cv2.rectangle(cv_img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), -1)
                r,g,b = dominant_clr
                cv2.rectangle(cv_img, (rect_x-rect_h, rect_y), (rect_x, rect_y + rect_h), (b,g,r), -1)
                cv2.rectangle(cv_img, (rect_x-rect_h, rect_y), (rect_x, rect_y + rect_h), (0,0,0), 3)
                cv2.putText(cv_img, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

                # Display the image
                cv2.imshow('Unique-img (ColorAnalysis)', cv_img)
                k = cv2.waitKey(0)
                if k==27:
                    break
        

    if not display:
        # Write the dataframe to a CSV file
        df.to_csv(config.csv_ptest, index=False)
    else:
        cv2.destroyAllWindows()


        


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Time it took to perform color-analysis is {time.time()-start} secs")