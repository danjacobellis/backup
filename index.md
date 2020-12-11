---
layout: default
title: Blog Post
tagline: Speech Separation using Audio-Visual Features
description: Blog post for EE 380L Data Mining Project
---

***Alexander Fiore, Alexander Phillips, Arivarasi Kesawaram, Dae Yeol Lee, Dan Jacobellis***

![Cocktail Party Effect](/img/cocktail.png)

[comment]: # (Abstract: 1-2 Paragraphs)


[comment]: # (Introduction & Background / Problem being addressed and why it’s important / Related work / approach and rationale / contribution or novel characteristics)
# 1. Introduction and Motivation
Imagine you are at a holiday party, or a gathering with friends and you are engaged in a discussion with some of your friends. A lot of people are talking simultaneously, but you are not paying attention to those conversations. But then, you hear someone say your name suddenly and you realize a group of people having a discussion behind you. Now you are unable to concentrate on the discussion you were having previously, because you are curious to know what the other people are talking about you. You weren’t deliberately eavesdropping on the group’s conversation, you just happened to hear your name. Is it even possible to unconsciously eavesdrop? 

![Cocktail Party Effect](/img/party_illustration.png)  

This scenario is what we call a ‘cock-tail party phenomenon’. It describes the ability to focus one’s listening attention on a single speaker among a mixture of conversations and background noises, ignoring other conversations. Our brain does this Audio-source separation all the time without us even realizing it. There is a lot that is unknown about the processes in the human auditory system that achieve this, and developing computational methods to replicate it remains an open problem. The binaural processing of the human auditory system in combination with visual cues allows us to effectively focus our attention on a single speaker when multiple speakers and other noise are present. When audio containing multiple speakers is recorded digitally and combined into a single channel (e.g. the recent presidential debates), the visual and spatial component of the acoustic field is lost, and it becomes much harder for a listener to understand the speakers individually. So, this is a difficult problem to solve in the world of signal processing.

In this project, we have built a Deep Neural Network (DNN) model, which is a joint audio-video separation model that can take in both auditory and visual features extracted from a video and decomposes the input mixed audio track into distinct output audio tracks, one for each speaker in the video. We train the model to estimate a time-frequency audio mask that separates two speakers in a single channel recording. This mask when applied to the input mixed audio, will filter out the desired individual track. 
This is what a high-level view of what the model looks like.

![Looking to listen](/img/looking-to-listen.PNG)
Source: Google Research, https://looking-to-listen.github.io

Over 5000 videos were retrieved from the AVspeech dataset and combined into pairs to form speech mixtures and ground truth labels. To our knowledge, this project is the first publicly available end-to-end implementation for this task.
But, why do we require the visual information or features if our end result is two separate audio tracks?

Videos are an increasingly common modality for speech transmission, whether it is a zoom meeting or a presidential debate. Unfortunately they usually contain only a single audio channel so spatial information  cannot be used for source separation. In general, separating speech from a single channel recording is an open problem, and our project focuses on using a statistical learning approach to solve it.
The importance of visual features could be better understood with the help of an example. Watch the first video clip. It is a short clip of two stand-up comedians performing at the same time. This is an example of the mixed input audio track to the model. (video 1)

The second clip is an example result of the model developed by Google Research using both auditory and visual features. (video 2) We can hear one speaker distinctly, even though both of them are talking at the same time. There is much less interference from the other speaker.This is because the visual component of human speech provides rich information about the acoustic signal that can be exploited for source separation. 

The third clip is the result of another model that uses blind source separation method or only audio features. (video 3) The blind method is overly aggressive in masking, resulting in the useful audio also cutting out and this encourages the use of secondary information like visual features or spatial information. Since recorded audio is increasingly accompanied by video (but not by multi-channel audio), exploiting visual information is more convenient and useful. Use of the visual signal is further motivated further by its known power, as evidenced by human lip reading.

Now let’s discuss some of the biggest challenges when it comes to developing a learning model for this problem. 

Firstly, video features are a lot to handle. There are several frames per second, and each frame has a lot of pixel information depending upon the quality of the video and there are three channels for each frame. You can imagine how massive the data would be if you tried to use raw video, so we need a way to reduce the dimensionality. Our approach is to use transfer learning of existing facial recognition networks like FaceNet to reduce the load for our problem.
Secondly, for the audio, we are less concerned with the raw size of the input, since it is more manageable, but we do need to put the audio in a form that is easily separable. For example you can imagine a signal consisting of two pulses that you could easily separate in the time domain, but you could also imagine a signal consisting of two tones that would be easy to separate in the frequency domain by first taking a discrete Fourier transform. Before attempting to train a model, it is necessary to put the audio input in some form that is easiest to separate.

## Related Work
To our knowledge, the current state of the art for audio-visual separation is a deep network model developed by Google Research which was later accompanied by the AVSpeech dataset. While its results are impressive, the implementation is proprietary and many of the details of the processing pipeline are left ambiguous. Regardless, it provides an excellent learning model architecture and introduces several novel techniques which we have adopted. To our knowledge, our implementation is the first publicly available, end-to-end audio-visual speech separation model.

## Overview of architecture
The model explored in our project is a deep neural network (DNN) trained using face embeddings generated by the FaceNet model and the mixture of audio streams from two videos. The output of the model is a time-frequency audio mask to be used for audio source separation. The adopted audio-visual network architecture consists of both convolutional and recurrent layers.

The visual pre-processing consists of (1) a face detection model, (2) rejection of videos with occluded faces (3) resampling of cropped faces to a uniform size, and (4) per-frame application of the FaceNet model to create a sequence of face embeddings. The audio pre-processing consists of (1) mixing the audio from pairs of videos containing speech, (2) applying a time-frequency transform, and (3) constructing the ideal mask to use as the ground truth in training.

YouTube videos referenced by URLs in the AVspeech dataset were downloaded and preprocessed in pairs to generate training data corresponding to a two-speaker scenario. The final model was trained on approximately 2500 mixtures from 5000 YouTube videos containing unique speakers

A variant of the structural similarity measure used in image quality assessment was used as the loss between the predicted and ideal audio masks during the training. The network was trained using a consumer GPU on a personal workstation iterating over the full dataset in several epochs. It produces speech separation masks which show some similarity to the ideal masks, but the error remains quite high judging by listening to the reconstructed audio streams.

# 2. Technical Approach
![flowchart_overall](/img/flowchart_overall.PNG)  
The figure above shows an overview of the preprocessed data inputs, the learning model, and predicted output or separating mask. In summary, the 1792 x 75 face embeddings from the output of the FaceNet model are fed into several convolutional layers with the constraint of having shared weights for each speaker’s face. In parallel, the STFT of the audio mixture is fed into additional convolution layers custom to the audio’s dimensions. Finally the audio and visual features are combined and passed through recurrent layers to exploit the temporal nature of the data. A separating mask is predicted as the final output layer for one speaker such that the second speaker’s mask can be derived from the complement of the predicted mask.

# 3. Dataset and Dataset Synthesis
The dataset used for the speech separation was a subset of Google’s [AVSpeech dataset](https://looking-to-listen.github.io/avspeech/download.html). In its raw form, the AVSpeech dataset provides a collection of YouTube links of three to ten second videos of single person speech. The AVSpeech data set contains over 280,000 random samples of videos of speakers with mixed languages with no interfering background noise. To build a training set for single channel speech separation the single speaker audio from the YouTube videos would need to be superimposed to synthesize a useful dataset. A spectrogram from the normalized mixed audio was then derived using a short-time fourier transform and concatenated with the visual feature vector. The following steps to generate a usable dataset can be described below.

* Extract (download and re-encode) raw audio/video data
  * youtube-dl to download data
  * ffmpeg to extract frames and audio stream
* Normalize Audio, Crop Images, and extract features from data
  * Construct time-frequency images
  * Apply FaceNet face detection model
  * Resample face images to uniform size
* Choose pairs of videos to create training data consisting of mixed speech
  * Discard videos with occluded faces
  * Construct ideal mask for each pair