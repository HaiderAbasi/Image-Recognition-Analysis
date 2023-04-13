import pandas as pd
from PIL import Image
import concurrent.futures

import os
import time
import config
import imagehash
import multiprocessing

# Define the function to calculate image hash
def calculate_hash(img_path):
    img = Image.open(img_path)
    img_hash = str(imagehash.average_hash(img))
    return img_hash, os.path.basename(img_path)

def main():

    # Get a list of image paths
    img_paths = [os.path.join(config.data_dir, filename) for filename in os.listdir(config.data_dir)]

    # Calculate hashes in parallel using concurrent.futures
    hashes = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(calculate_hash, img_path) for img_path in img_paths]
        for future in concurrent.futures.as_completed(futures):
            img_hash, filename = future.result()
            if img_hash in hashes:
                hashes[img_hash].append(filename)
            else:
                hashes[img_hash] = [filename]

    # Initialize an empty list to store the unique image ids and the number of duplicates
    data = []

    # Iterate over the dictionary of image hashes
    for img_hash, filenames in hashes.items():
        # Store the unique image id in the list
        data.append({'image_id': filenames[0], 'num_duplicates': len(filenames)-1})

    # Create the dataframe from the list of data
    df = pd.DataFrame(data, columns=['image_id', 'num_duplicates'])
    
    # Write the dataframe to a CSV file
    df.to_csv(config.csv_ptest, index=False)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    start = time.time()
    main()
    print(f"Time it took to process find duplicates is {time.time()-start} secs")
