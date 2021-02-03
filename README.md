# Single Image Reflection Removal with Edge Guidance, Reflection Classifier, and Recurrent Decomposition (WACV 2021 accepted)
PyTorch implementaton of the following paper. In this paper, we propose a novel model with auxiliary techniques to tackle the problem of single image reflection removal. 
<div align=center><img height="300" src="https://github.com/JennaChangY/Reflection-Removal-with-Auxiliary-Techniques/blob/main/teaser.png"/></div>
Given a reflection contaminated input image (the first column), our method aims to decompose the reflection layer (the last column) and generate the reflection-free transmission
layer (the third column), which must be quite similar to its corresponding groundtruth (the second column).  

## Paper
[Single Image Reflection Removal with Edge Guidance, Reflection Classifier, and Recurrent Decomposition](https://people.cs.nctu.edu.tw/~walon/publications/chang2021wacv.pdf)  
Ya-Chu Chang, Chia-Ni Lu, Chia-Chi Cheng, [Wei-Chen Chiu](https://walonchiu.github.io/)  
IEEE Winter Conference on Applications of Computer Vision (WACV), 2021.  

Please cite our paper if you find it useful for your research.  
```
@inproceedings{chang21wacv,
 title = {Single Image Reflection Removal with Edge Guidance, Reflection Classifier, and Recurrent Decomposition},
 author = {Ya-Chu Chang and Chia-Ni Lu and Chia-Chi Cheng and Wei-Chen Chiu},
 booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
 year = {2021}
}
```

## Installation
* This code was developed with Python 3.1.7 & Pytorch 1.0.1 & CUDA 9.0.
* Other requirements: numpy, cv2
* Clone this repo
```
git clone https://github.com/JennaChangY/Reflection-Removal-with-Auxiliary-Techniques.git
cd Reflection-Removal-with-Auxiliary-Techniques
```

## Testing
Download our pretrained models from [here](https://drive.google.com/drive/folders/1fZPnxjmI_2auJVjIc6jZAhDBWvdmTZuo?usp=sharing) and put them under `weights/`.  
Run the sample data provided in this repo:
```
python test.py
```
Run your own data:
```
python test.py --data_dir YOUR_DATA_PATH
               --save_dir YOUR_SAVE_PATH
```

## Training
Create the synthetic training data,  
 - images from 2012 PASCAL VOC training images with the generation method from [CEILNet](https://github.com/fqnchina/CEILNet/).  
 - images on Flickr with the generation method from [Zhang et al.](https://github.com/ceciliavision/perceptual-reflection-removal)    
Download the real-world training data obtained from [Zhang et al.](https://github.com/ceciliavision/perceptual-reflection-removal) and apply typical data augmentation techniques (flipping and random cropping).  
Or you could download whole training dataset [here](https://drive.google.com/drive/folders/1ogX2XORPgLhjj3BThGo4pKCCcG0A-fCf?usp=sharing) and put them under `train_data/`.  
Then you can directly run the training code:
```
python train.py
```
