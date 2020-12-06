# Single Image Reflection Removal with Edge Guidance, Reflection Classifier, and Recurrent Decomposition (WACV 2021 accepted)
PyTorch implementaton of the following paper. In this paper, we propose a self-supervised end-to-end model to generate the timelapse
video from a single image and a reference video.  
<div align=><img height="200" src="https://github.com/angelwmab/Time-Flies/blob/master/figure/teaser.gif"/></div>

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

## Example Results
![img](https://github.com/angelwmab/Time-Flies/blob/master/figure/qualitative.png)

## Installation
* This code was developed with Python 3.1.7 & Pytorch 1.0.1 & CUDA 9.0.
* Other requirements: numpy, cv2
* Clone this repo
```
git clone https://github.com/JennaChangY/Reflection-Removal-with-Auxiliary-Techniques.git
cd Reflection-Removal-with-Auxiliary-Techniques
```

## Testing
Download our pretrained models from [here](https://drive.google.com/drive/folders/1fZPnxjmI_2auJVjIc6jZAhDBWvdmTZuo?usp=sharing) and put them under `models/`.  
Run the sample data provided in this repo:
```
python test.py
```
Run your own data:
```
python test.py --data_dir YOUR_REF_VID_FRAME_PATH
               --target_img_path YOUR_TARGET_IMG_PATH
```

## Training
Download the webcamclipart dataset [here](http://graphics.cs.cmu.edu/projects/webcamdataset/) and put them under `webcamclipart/`.  
Download the segmentation maps of each scene [here](https://drive.google.com/drive/folders/1_RGhDdLSpdrb_bk0x-EkXz9Jmhm3AQHY?usp=sharing) and put them under `segmentations/`.  
Then you can directly run the training code:
```
python train.py
```
Train the model with your own dataset:
```
python train.py --data_dir YOUR_REF_VID_DATASET
```
