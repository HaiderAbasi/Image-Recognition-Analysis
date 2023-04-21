import numpy as np
import torch
import torchvision.transforms as t
import cv2

from src.milestone_4.alexnet import KitModel as AlexNet
from src.milestone_4.vgg19 import KitModel as VGG19
from PIL import Image



def compute_img_sentiment(image_path, model_type='hybrid_finetuned_all'):
    transform = t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Lambda(lambda x: x[[2, 1, 0], ...] * 255),  # RGB -> BGR and [0,1] -> [0,255]
        t.Normalize(mean=[116.8007, 121.2751, 130.4602], std=[1, 1, 1]),  # mean subtraction
    ])

    # Load the model
    model = AlexNet if 'hybrid' in model_type else VGG19
    model = model(f'data/models/{model_type}.pth').to('cpu')
    model.eval()

    # Load the image
    image = Image.open(image_path)

    # Transform the image and add a batch dimension
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Forward pass through the model
        output = model(x).cpu().numpy()[0]
    
    sentiments = ['negative', 'neutral', 'positive']
    prediction = np.argmax(output)
    img_sentiment = sentiments[prediction]

    # Return the predicted sentiment (0: negative, 1: neutral, 2: positive)
    return img_sentiment


if __name__ == '__main__':
    models = ('hybrid_finetuned_fc6+',
          'hybrid_finetuned_all',
          'vgg19_finetuned_fc6+',
          'vgg19_finetuned_all')

    model_type = models[1]
    image_path = r"src\milestone_4\haha.jpg"
    img_sentiment = compute_img_sentiment(image_path, model_type)
    img = cv2.imread(image_path)
    cv2.putText(img,img_sentiment,(30,30),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
    cv2.imshow("Image (Sentiment Analysis)",img)
    cv2.waitKey(0)