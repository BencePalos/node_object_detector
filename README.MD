Small object detection model with a some extra miscellaneous things for the future.
Run in node.js locally, parses images(jpg, jpeg, png) in the images directory and uses a tfjs converted MobileNetV2 model to give you predictions and their probability on what it thinks is on the image.

Here's a link of the conversion colab:
https://colab.research.google.com/drive/1vnQXSgJ0aqCAM2iIT3jypdiF45a_nN0D#scrollTo=dTdTcj2r5cyT

With the currently running mobileNetPredict() function it's capable of recognizing 1000 different kinds of objects, and it will give you the most probable result only.

Misc.
It has a cocoDetect() function which utilizes the pre-traind COCO-SSD model that is capable of recognizing 80 different kinds of objects.

It has a python script that is usable for training models, and has been used to train the flower_model in the misc directory.
This produces saved_models which are currently not usable in tfjs due to C++ only operations, and for now fail the conversion because of the same issue.
