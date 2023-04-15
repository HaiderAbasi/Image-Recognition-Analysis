from PIL import Image
import pandas as pd

import os
import time
import config
import imagehash

start = time.time()

# Initialize an empty dictionary to store the image hashes
hashes = {}

# Iterate over all the images in the folder
for filename in os.listdir(config.data_dir):
    # Read the image file
    img_path = os.path.join(config.data_dir, filename)
    # Calculate the image hash using ImageHash
    img = Image.open(img_path)
    img_hash = str(imagehash.average_hash(img))
    # Store the image hash in the dictionary
    if img_hash in hashes:
        hashes[img_hash].append(filename)
    else:
        hashes[img_hash] = [filename]

# Initialize an empty list to store the unique image ids and the number of duplicates
data = []
# Iterate over the dictionary of image hashes
for img_hash, filenames in hashes.items():
    # Iterate over the list of filenames and add the duplicates to the CSV file
    for i in range(len(filenames)):
        duplicate_of = filenames[0] if i > 0 else "-"
        data.append({'image_id': filenames[i], 'duplicate_of': duplicate_of})

# Create the dataframe from the list of data
df = pd.DataFrame(data, columns=['image_id', 'duplicate_of'])

# Write the dataframe to a CSV file
df.to_csv(config.csv_test, index=False)

print(f"Time it took to process find duplicates is {time.time()-start} secs")