# Sunglasses of Invisibility

Augmented reality project which uses Mask R-CNN. A person wearing sunglasses will disappear. If there is an object in front of the person with the sunglasses, the object will stay in focus. The object will have a levitation effect. Only objects that are part of the [COCO "Things" dataset](https://github.com/nightrome/cocostuff#labels) with segmentation annotations are detectable in this project.

## Demo

<a href="https://www.youtube.com/watch?v=QhSDOCkkeXA" target="_blank"><img src="https://img.youtube.com/vi/QhSDOCkkeXA/0.jpg" 
alt="Sunglasses of Invisibility Demo YouTube" width="240" height="180" border="10" /></a>

## Inspiration

The idea for this project came from the movie <i>Big Daddy</i> with Adam Sandler. There's a scene where Adam Sandler gives the boy Julian a pair of sunglasses and tells him that he will become invisible if he wears them.

<a href="https://www.youtube.com/watch?v=V2zEfKnf2iw" target="_blank"><img src="https://img.youtube.com/vi/V2zEfKnf2iw/0.jpg" 
alt="Big Daddy Sunglasses Scene YouTube" width="240" height="180" border="10" /></a>

## References

This project was created by leveraging the code provided by [Matterport, Inc.](https://github.com/matterport/Mask_RCNN). The structure of this project was copied from Matterport's [demo Jupyter Notebook](https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb). They also give a great explanation of how instance segmentation works in this [tutorial](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46).

This YouTube [video](https://www.youtube.com/watch?v=2TikTv6PWDw) by Mark Jay was very helpful in explaining how to install all the necessary packages. 

## Technical Specs

Python 3.6.3 <br/>
OpenCV-python 3.4.0.12 <br/>
TensorFlow 1.8 <br/>
Keras 2.1.6 <br/>

## Getting Started

Watch Mark Jay's YouTube [video](https://www.youtube.com/watch?v=2TikTv6PWDw) and install all the necessary packages before you run any code from this repository. Mark also shows you how to download the weights that will be used for the instance segmentation model. This file is called `mask_rcnn_coco.h5`. Do not rename it. After downloading it, place it in the same folder as the `tutorial_single_image` Notebook.

Open the [tutorial_single_image](https://github.com/michaelnation26/sunglasses_of_invisibility/blob/master/tutorial_single_image.ipynb) Notebook. You can simply run the entire Notebook. If you did not install all of the required packages, the first cell will display an error. This Notebook will visually show how instance segmentation was used to crop out a person in an image.

The [invisibility_on_video.py](https://github.com/michaelnation26/sunglasses_of_invisibility/blob/master/invisibility_on_video.py) file is used to read in a prerecorded video and output another video with the invisibility effects. If you want to use your own prerecorded video, place it in the `videos` folder and change the video filename on line 25 in the `invisibility_on_video.py` file.

## Additional Notes

The first frame in the prerecorded video is saved and used as the "background" image to replace the person's image pixels. Due to this, the first frame should not have any people in it.

The structure of the `invisibility_on_video.py` file allows for the invisibility processing to happen in real-time but the amount of time it takes to process a single image on my MacBook Pro using a CPU is about 20 seconds. I have an old MacBook Pro and the GPU card isn't supported by NVIDIA. Facebook AI Research was able to process 5 fps using Mask R-CNN.

If you would like to attempt to convert `invisibility_on_video.py` to real-time image processing, change line 9 in `invisibility_on_video.py` to <i>video = cv2.VideoCapture(0)</i>. This will allow you to stream video from your laptop's camera.