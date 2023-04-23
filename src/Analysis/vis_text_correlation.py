
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os
import ast
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
warnings.filterwarnings("ignore")
from src.milestone_4.visual_sentiment_analysis import SentimentAnalyzer
s_analyzer = SentimentAnalyzer()


def to_clr(dominant_clr_str):
    dominant_clr =  ast.literal_eval(dominant_clr_str)
    clr = s_analyzer.map_to_dominant_color(dominant_clr)
    return clr

sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

def int_format(pct,sries):
    return '{:.0f}'.format(pct * sum(sries.values) / 100)

#---------------------------------------------------------------------------------------------------------#

# load the csv file into a pandas DataFrame
df = pd.read_csv(config.csv_display)
print("===================================================================================================")
print("1) Original DataFrame:")
print(df.head(n=3))

# select only the rows where duplicate_of is "-"
unique_images = df[df['duplicate_of'] == '-']
print("===================================================================================================")
print("2) DataFrame with duplicate rows removed:")
print(unique_images.head(n=3))

# add a new column to calculate the total number of people
unique_images['People Count'] = unique_images['Adults'] + unique_images['Children']
print("===================================================================================================")
print("3) DataFrame with People Count column added:")
print(unique_images.head(n=3))

# filter out the focus subjects that are equal to '-'
unique_images = unique_images[unique_images['focus_subject'] != '-']
print("===================================================================================================")
print("4) DataFrame with focus subjects filtered out:")
print(unique_images.head(n=3))

# Count the number of duplicates for each image
focus_subject_counts = unique_images['focus_subject'].value_counts()
print("focus_subject_counts = ",focus_subject_counts)

# Map the count of each focus object to the size of the bubble
size_map = {focus_subject: count for focus_subject, count in zip(focus_subject_counts.index, focus_subject_counts)}
unique_images['focus_obj_occurence'] = unique_images['focus_subject'].map(size_map)


print("===================================================================================================")
print("5) DataFrame with focus_obj_occurence column added:")
print(unique_images.head(n=3))

unique_images['dominant_clr'] = unique_images['dominant_clr'].apply(to_clr)
print("===================================================================================================")
print("5) DataFrame with colors replaced with names:")
print(unique_images.head(n=3))

# create the scatter plot with bubbles
plt.figure(figsize=(10, 8), tight_layout=True)
custom_palette = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
sns.scatterplot(data=unique_images, x='People Count', y='focus_subject',size = 'focus_obj_occurence', sizes=(30, 300), alpha=0.7, hue='image_sentiment',palette=custom_palette,style='text_present')

# set the title and axis labels
plt.title('Scatter Plot with Bubbles')
plt.xlabel('Number of People in the Scene')
plt.ylabel('Focus Object')


# display the plot
plt.show()