# rppg_demo

<br/>
This repository contains code for a livestream demo of rppg based on this paper https://arxiv.org/pdf/2504.01774 and the code base at https://github.com/KegangWangCCNU/ME-rPPG/tree/main . The face detection is formed using Mediapipe (https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) and the face detection model used is BlazeFace. The input to the code is a livestream video from an integrated web cam and the output is the heart rate, bvp, and signal quality. At the end of the session, the average heart rate and signal quality is displayed. Code from the Mediapipe face detection guide has been used (https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python#live-stream).
<br/><br/>

## Instructions for setup:

<br/>
This code was tested on Ubuntu 22.04 and python 3.10. Please follow the instructions below for setup.
<br/><br/>

```
 Clone the repository and go to the repository directory.
```

```
python3 -m venv rppgvenv
```

```
source rppgvenv/bin/activate
```

```
python3 install -r requirements.txt
```


## Instructions for Inference:

<br/>
An integrated webcam is required on the laptop. All the pretrained weights are already available in the repository.
<br/><br/><br/>

```
 python3 predict.py
```

<br/>

## Results 



## References

<br/>
[1] Wang K, Tang J, Fan Y, Ji J, Shi Y, Wang Y. Memory-efficient low-latency remote photoplethysmography through temporal-spatial state space duality. arXiv preprint arXiv:2504.01774. 2025 Apr 2.

[2] https://github.com/KegangWangCCNU/ME-rPPG/tree/main

[3] https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

[4] https://github.com/aliyun/NeWCRFs](https://github.com/ubicomplab/rPPG-Toolbox
