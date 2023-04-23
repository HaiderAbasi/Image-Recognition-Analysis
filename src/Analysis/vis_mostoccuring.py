
import pandas as pd
import matplotlib.pyplot as plt
import config
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import seaborn as sns


sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

def int_format(pct):
    return '{:.0f}'.format(pct * sum(top_10.values) / 100)

#---------------------------------------------------------------------------------------------------------#

# load the csv file into a pandas DataFrame
df = pd.read_csv(config.csv_display)

# select only the rows where duplicate_of is "-"
unique_images = df[df['duplicate_of'] == '-']

# count the number of duplicates for each unique image
duplicate_counts = {}
for image_id in unique_images['image_id']:
    duplicate_counts[image_id] = len(df[df['duplicate_of'] == image_id])

# sort the counts in descending order and select the top 10
top_10 = pd.Series(duplicate_counts).sort_values(ascending=False)[:5]

# create a series for the rest of the values
rest_sum = sum(duplicate_counts.values()) - sum(top_10.values)
rest = pd.Series(rest_sum, index=['Rest'])

# concatenate the top 10 series with the rest series
top_10 = pd.concat([top_10, rest])



# plot the top 10 counts as a pie chart
fig, ax = plt.subplots(figsize=(8, 8))  # set the figure size
# set aspect ratio to equal
ax.set_aspect('equal')
colors = sns.color_palette('pastel')
explode = [0.0] * len(top_10)
explode[2] = 0.2


#'%1.1f%%'
wedges,labels,_ = ax.pie(top_10.values, labels=top_10.index, colors=colors, explode=explode, autopct='%1.1f%%',shadow=True,labeldistance=1.2)

# Set the size of the thumbnail
thumbnail_size = 0.15

for label, wedge in zip(labels, wedges):
    # Get the image path for the current label
    image_path = label.get_text()  # assuming the image file name is the same as the label
    if image_path =="Rest":
        continue
    image_path = os.path.join(config.data_dir, image_path)

    # Load the image and create an offset box with it
    image = plt.imread(image_path)
    
    image_box = OffsetImage(image, zoom=thumbnail_size)
    
    # Set the position of the image box just below the label
    image_pos = (label.get_position()[0], label.get_position()[1] )
    
    # Create an annotation box with the image box and add it to the plot
    image_annotation = AnnotationBbox(image_box, image_pos, frameon=False)
    ax.add_artist(image_annotation)
    # Set the label's alpha value to 0
    label.set_alpha(0)    
    
ax.set_title('Top 10 Unique Images with the Highest Number of Duplicates', fontsize=14, y=1.1)  # set the title and font size
plt.show()