from colorthief import ColorThief
import numpy as np


def get_dominant_color(img_path):
    
    color_thief = ColorThief(img_path)
    # get the dominant color
    colors_to_find = 2
    quality = 1 # 1 (highest) -> 10 (lowest)
    dominant_color = color_thief.get_palette(colors_to_find, quality)[0]
    return dominant_color
        
def identify_sat_brightness_levels(img):
    hsv = img.convert('HSV')
    _, sat, value = hsv.split()

    # Compute the average saturation and brightness levels
    avg_saturation = np.mean(sat)
    avg_brightness = np.mean(value)
    
    # Determine the saturation level
    if avg_saturation < 64:
        saturation_level = 'Low'
    elif avg_saturation < 192:
        saturation_level = 'Medium'
    else:
        saturation_level = 'High'
        
    # Determine the brightness level
    if avg_brightness < 64:
        brightness_level = 'Low'
    elif avg_brightness < 192:
        brightness_level = 'Medium'
    else:
        brightness_level = 'High'
        
    # Return the saturation and brightness levels
    return saturation_level, brightness_level

        
        
def perform_color_analysis(img,img_path):
    
    dominant_color = get_dominant_color(img_path)
    

    
    sat_level,brightness_level = identify_sat_brightness_levels(img)
    
    
    return dominant_color,sat_level,brightness_level
        

