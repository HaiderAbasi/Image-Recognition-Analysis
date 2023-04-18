import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class AgeClassifier:
    def __init__(self, model_path = r'data\models\mobilenet_v2_adult_child_classifier.h5'):
        self.model = load_model(model_path)
        self.class_names = {0: 'adults', 1: 'children'}

    def predict(self, image_path):
        if isinstance(image_path, str):
            img = image.load_img(image_path, target_size=(150, 150))
        else:
            img = image_path.resize((150, 150))  # resize the image

        #plt.imshow(img)
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = self.model.predict(images, batch_size=10,verbose =0)
        score = tf.nn.sigmoid(classes[0])

        if classes[0] > 0.5:
            return 1
        else:
            return 0
        



if __name__ == "__main__":
    classifier = AgeClassifier()
    class_id = classifier.predict(r"src\milestone_3\test_imgs\children\0.jpg")
    print(f"class_id = {class_id}")
