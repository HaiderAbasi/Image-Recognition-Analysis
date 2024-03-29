# Config file for Image Recognition Analysis Repo.

common_case = "_full"
display = False
display_face_analysis = False

debug = False
# Main Variables


# Models
deep_sentiment_analyzer = "hybrid_finetuned_fc6+" # "hybrid_finetuned_all"


# Testing Variables
case = "" if common_case == "" else common_case
data_dir = r'data\raw\test' + case


# Intermediate results directory
processed_dir = 'data/processed/'
csv_test = processed_dir + f"image_duplicates{case}_normal.csv"
csv_ptest = processed_dir + f"image_duplicates{case}.csv"


# Visualization Variables
case_display = "" if common_case == "" else common_case
csv_display = processed_dir + f"image_duplicates{case_display}.csv"
