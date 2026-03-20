# rppg_demo

<br/>
This repository contains code for a livestream demo of rppg based on the paper "Memory-efficient low-latency remote photoplethysmography through temporal-spatial state space duality" (https://arxiv.org/pdf/2504.01774). The face detection is done using Mediapipe face detection and the face detection model used is BlazeFace. The input to the code is a livestream video from an integrated web cam and the output is the heart rate, bvp, and signal quality. At the end of the session, the average heart rate and signal quality is displayed.
<br/><br/>

## Instructions for setup:

<br/>
This code was tested on Ubuntu 22.04 and python 3.10. Please follow the instructions below for setup.
<br/><br/>

```
 Clone the repository and go to the repository directory.
```
<br/>
Create a virtual environment.
<br/><br/>

```
python3 -m venv rppgvenv
```
<br/>

```
source rppgvenv/bin/activate
```

<br/>
Install the requirements.
<br/><br/>

```
python3 install -r requirements.txt
```
<br/>

## Instructions for Inference:

<br/>
An integrated webcam is required on the laptop. All the pretrained weights are already available in the repository.
<br/><br/><br/>

```
 python3 predict.py
```

<br/>

## Results 

<br/>
Please watch a demo video using the link below.
<br/><br/>

https://www.dropbox.com/scl/fi/fa8qhlk0lkqjw8zfh4pla/result.webm?rlkey=46ufq3fhcopena57ivay129rf&st=mdg6m56x&dl=0

<br/>

## Acknowledgements



<br/>
The code is in this repository is based on the code in this (https://github.com/KegangWangCCNU/ME-rPPG/tree/main) repository. The pretrained weights have also been obtained from the same. The code for face detection has been taken from Mediapipe face detection guide (https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector). The pretrained Blazeface model has been obtained from the same. 
<br/><br/>


## References

<br/>
[1] Wang K, Tang J, Fan Y, Ji J, Shi Y, Wang Y. Memory-efficient low-latency remote photoplethysmography through temporal-spatial state space duality. arXiv preprint arXiv:2504.01774. 2025 Apr 2.
<br/><br/>

[2] https://github.com/KegangWangCCNU/ME-rPPG/tree/main

[3] https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

[4] https://github.com/aliyun/NeWCRFs](https://github.com/ubicomplab/rPPG-Toolbox

