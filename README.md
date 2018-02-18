# facial_emotion_recognition
 
This repository contain facial expression using cnn, you can also use webcam as realtime facial expression detection. 

## Dataset 
for dataset I use FER 2013 from kaggle, you can download here
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge 

## Dependency 
- Keras
- Tensorflow
- Opencv

## Pretrained model
For keras model you can download here
https://drive.google.com/file/d/0B6yZu81NrMhSV2ozYWZrenJXd1E/view?usp=sharing 
and put in keras_model file

##How to use

1. For real time facial emotion recognition you can execute command: ```python realtime_facial_expression.py``` 

2. For detecting the facial expression in image you can execute the command ```
python image_test.py tes.jpg```

#### Image testing example 
![result_emotion_detection_app](https://user-images.githubusercontent.com/12840374/36295924-8380b372-1310-11e8-8646-2157f6ea98f5.jpg)

#### Demo Image example

![b08394_11_42](https://user-images.githubusercontent.com/12840374/36353364-bfe558a4-14eb-11e8-9649-7a421bea7772.png)


## Credit
This code credit goes to [adamaulia](https://github.com/adamaulia). I've merely created a wrapper to get people started.