# MFNet

This repository contains my work done on the problem of masked face recognition.
I have used the Labelled Faces in the Wild (LFW) Dataset.
Two different custom datasets were being created from the LFW dataset using MaskTheFace[https://github.com/aqeelanwar/MaskTheFace] Tool.
The InceptionResnetV1 model pretrained on vggface2 and casia-webface datasets was being used for experiments.
All the experiments were done under two different configurations with rest of the settings being same to have better grounds for comparison and evaluation purpose.
Along with the usual classification setting for any image classification problem , we further utilize the triplet loss to train our models in a self supervised manner.


`Dependencies`

- pytorch
- torchvision
- facenet-pytorch
- dlib
- matplotlib
- seaborn
- numpy
- pandas
