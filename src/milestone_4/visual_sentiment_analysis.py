import numpy as np
import colorsys

class SentimentAnalyzer:
    
    def __init__(self):
        pass
        
    emotion_to_numeric = {"angry": -0.9,"disgust": -0.8 , "fear": -0.7, "happy": 0.9,
                          "sad": -0.8  ,"surprise": 0.7 ,"neutral": 0.0}
    # Set weights for each feature based on their importance
    brightness_weight = 0.2
    saturation_weight = 0.1
    emotion_weight = 0.5
    color_weight = 0.2
    
    @staticmethod
    def map_to_dominant_color(clr, case="rgb"):
        if case == "bgr":
            rgb_color = (clr[2],clr[1],clr[0])
        else:
            rgb_color = clr
        
        # Convert RGB to HSV
        hsv_color = colorsys.rgb_to_hsv(*rgb_color)

        dominant_colors = {
            'Red': (0, 15),
            'Orange': (16, 45),
            'Yellow': (46, 60),
            'Green': (61, 150),
            'Turquoise': (151, 195),
            'Blue': (196, 270),
            'Purple': (271, 290),
            'Pink': (291, 345),
            'Brown': (346, 15),
            'Gray': (0, 0)
        }

        color_name = ''
        max_value = 0
        for name, (lower, upper) in dominant_colors.items():
            # Check if hue value is within the dominant color range
            if lower <= hsv_color[0] * 360 <= upper:
                value = hsv_color[2] * hsv_color[1]
                if value > max_value:
                    max_value = value
                    color_name = name

        return color_name
    
    
    @staticmethod
    def __map_color_to_sentiment(color_name):
        sentiment_values = {
            'Red': 1.0,
            'Orange': 0.8,
            'Yellow': 0.6,
            'Green': -0.6,
            'Turquoise': -0.8,
            'Blue': -1.0,
            'Purple': -0.6,
            'Pink': 0.8,
            'Brown': -0.2,
            'Gray': 0.0
        }
        return sentiment_values.get(color_name, 0.0)


    def analyze(self,mean_brightness, mean_saturation, human_emotion, dominant_color):
        
        brightness_sentiment = np.interp(mean_brightness,[0,255],[-1,1])
        saturation_sentiment = np.interp(mean_saturation,[0,255],[-1,1])
        
        color_name = self.map_to_dominant_color(dominant_color)
        clr_sentiment = self.__map_color_to_sentiment(color_name)
        

        if human_emotion !="-":
            human_sentiment = self.emotion_to_numeric[human_emotion]
            # Calculate overall sentiment score based on feature weights
            sentiment_score = (self.brightness_weight * brightness_sentiment) + (self.saturation_weight * saturation_sentiment) + \
                              (self.emotion_weight * human_sentiment)         + (self.color_weight * clr_sentiment)
        else:
            # No humans present, Compute image sentiment based on image characteristics
            sentiment_score = (0.4 * brightness_sentiment) + (0.4 * saturation_sentiment) + \
                              (0.2 * clr_sentiment)
            human_sentiment = 0
            
        # print(f"\nbrightness_sentiment = {brightness_sentiment:.1f}")
        # print(f"saturation_sentiment = {saturation_sentiment:.1f}")
        # print(f"human_sentiment = {human_sentiment}")
        # print(f"clr_sentiment = {clr_sentiment}")

        
        # Determine sentiment based on sentiment score
        if sentiment_score > 0.2:
            sentiment = "Positive"
        elif sentiment_score < -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return sentiment,color_name