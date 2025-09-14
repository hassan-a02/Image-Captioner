# Image Captioner (Flickr8k)

This university project implements an image captioning system using deep learning (PyTorch) on the [Flickr8k dataset](https://forms.illinois.edu/sec/1713398). The model generates captions for images by combining a pretrained ResNet-50 encoder and an LSTM decoder. The captions the model generates are quite poor, however it produces somewhat accurate captions despite limited training, data and computational resources.

## Dependencies for Server

- Flask
- Pillow
- PyTorch
- Torchvision

## Usage

**Evaluation Only**

- Run the flask server (src/backend/backend.py)
- Go to localhost:5000 on your browser
- Upload an image