# Android Audio Speech Enhancement App

## Overview
This project contains an Android app that demonstrates how to use the ONNX Runtime to perform audio recognition using a pre-trained model. The app records a 1 second audio clip, reads the clip into a tensor, and passes it through the pre-trained model to obtain a prediction. The app then displays the prediction, as well as the inference time.


## Using the App


When you run the app, you will see a simple interface with a record button and a text view. To use the app, follow these steps:

1. Press the record button to start recording.

2. Speak into the microphone for 1 second.

3. The app will automatically stop recording after 1 second.

4. Press the Predict button to perform audio recognition on the recorded clip.

5. The app will display the prediction and the inference time in the text view.

## Understanding the Code

The app consists of a single activity, MainActivity, which contains the code for recording audio, reading the audio clip into a tensor, and performing inference using the pre-trained model.


The code is divided into several functions:

- onCreate: This function is called when the activity is created. It initializes the views and creates an instance of the ONNX Runtime environment and session.

- readWavFileToTensor: This function takes a WAV file as input and returns a 1 second tensor of floating point values. The function reads the WAV file header to obtain the sample rate and number of channels, and then reads the audio data into a byte array. The function converts the byte array into a float array by dividing each 16-bit sample by Short.MAX_VALUE. The function returns the first 16000 samples of the float array (corresponding to the first second of audio).

- readTXT: This function reads a text file containing a random tensor and returns the tensor as a float array.

- createORTSession: This function creates an instance of the ONNX Runtime session using the pre-trained model.

- runFilePrediction: This function takes a tensor of audio data as input and returns the output of the pre-trained model as a float array. The function creates an input tensor from the audio data and passes it to the ONNX Runtime session. The function also measures the inference time and includes it in the output float array.

- runPrediction: This function is similar to runFilePrediction, but instead of taking a tensor of audio data as input, it generates a random tensor and passes it to the ONNX Runtime session.
