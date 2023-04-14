# Image Analysis for Academic Study


## Overview

This project involves analyzing a dataset of approximately 25,000 images for an academic study. 

- The main objective is to identify the number of unique images and the number of duplicates in the dataset. 
- Additionally, the project aims to extract several features from the images, which include:

  - Dominant color
  - Saturation
  - Brightness
  - Presence of humans
  - Age category of humans
  - Facial expressions
  - General sentiment

## Installation

To run the code in this repository, you will need to have Python 3 installed on your machine. You can install Python by following the instructions on the official [Python website](https://www.python.org/downloads/).

Next, you will need to install the required Python packages. You can do this by running the following command in your terminal:
> Protip: To ensure efficient development and avoid conflicts between different Python projects, it is recommended to install required modules inside a virtual environment (virtualenv) and run Visual Studio Code (VS Code) using the virtualenv's Python interpreter. 

```
pip install -r requirements.txt
```


## Usage

To run the analysis on the dataset, simply run the `a_find_uniques_parallel.py` script in the `src/milestone_1_find_unique_images`. 

Once the analysis is complete, a CSV file called `image_duplicates_{case}.csv` will be generated in the `data/processed`, which contains the image IDs and their corresponding feature values.

### Visualization
- To visualize the unqiue images and their # of duplicates. 
  - Simply run `a_display_unique_imgs.py` after generating the `.csv` file by the above step.

## Repo Structure

```
Image-Recognition-Analysis
|
|
├───src                                 >     Code Scripts for each milestones
│  └──milestone_1_find_unique_images       
|
|
├───config.py                           >     Control variables for test + disp
|
|
├──data                                 >     Input(raw) data + intermediate results (processed)
│  ├──processed
│  └──raw
│     ├──test
│     └──test_full
|
├──results                              >     Complete output for each milestone (.csv + timing + visualization)
│  └──milestone_1_find_unique_images
|
└──Readme.MD                            >     Self Explanatory XD
```

## Acknowledgements

This project was developed as part of an academic study and was made possible by the following resources:

- The [OpenCV](https://opencv.org/) library for image processing.
- The [imagehash](https://github.com/JohannesBuchner/imagehash) library for Perceptual Image Hashing.

## License

This project is licensed under the BSD 2-Clause License - see the `LICENSE` file for details.