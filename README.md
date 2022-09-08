# Divide & Conquer Track Analyzer

## 1. What Is This For?
This repository offers a tool to perform divide & conquer analyses of audio signals with keras models. It was built for music inputs (which is where "track" comes from), but is applicable to any kind of audio signal like speech or environmental sounds.

### 1.1. Divide & Conquer?
Let's say we want to make a prediction about an attribute of an audio signal. Divide & Conquer, in the context of audio machine learning, refers to the following strategy:
1. Divide the signal into multiple shorter segments.
2. Train a machine learning model to correctly predict the attribute on a segment-level.
3. Use the model to make predictions for all segments of an audio signal.
4. Aggregate the individual predictions to a global prediction for the full signal.

This method has several benefits, especially when working with small datasets:
* More training data
* If overlap is used, serves as a form of data augmentation
* Potentially smaller model
* Can be used to predict signals of all lenghts

This module assumes a model on smaller segments is already trained (See my audio processing pipeline [SLAPP](https://github.com/MaxHilsdorf/single_label_audio_processing_pipeline)). This repository helps you to make predictions for new audio signals using the divide & conquer strategy.

### 1.2. Does It Generalize to My Task?
If you have a keras model trained on image representations (!) of short audio snippets (like spectrograms or MFCCs) and have an MP3 file you want to make a divide & conquer based prediction for, it is highly likely that you can apply this tool to your problem. It is built to allow you to add your own functionalities for feature extraction or data processing.

### 1.3. What Is Currently Implemented?
Currently, the track analyzer has all the functionality to predict with CNN or CRNN models which where trained on mel spectrograms with values normalized to the interval [0+$\Delta$,1+$\Delta$]. However, it is very easy to add your own functionalities, as laid out in section 4.


## 2. Requirements
Again, this tool currently only works with keras models which are used to predict attributes of MP3 files. Additionally, the following python libraries are needed:
* Numpy
* Pydub
* Librosa
* Tensorflow

Further, the audio codec [FFmpeg](https://ffmpeg.org/download.html) must be installed on your system.


## 3. How to Use?

Essentially, you can just customize the ```main.py``` file to your liking. After you have set your model and mp3 path, you are going to need suitable implementations of a ```FeatureExtractor``` and a ```DataProcessor```. The ```FeatureExtractor``` takes an audio signal and computes the feature of interest (e.g. a mel spectrogram). The ```DataProcessor``` takes a dataset of features and performs processing steps like normalization or the swapping of axes. You may have to write your own feature extractor and data processor. However, it is very easy to implement your own functionalities.

## 4. How to Add Functionalities?

If you want to add your own ```FeatureExtractor```, e.g. one to extract MFCCs from an audio signal, you will have to implement a class which inherits from the ```FeatureExtractor``` class in the ```feature_extractor.py``` module. Here is how you could go about that:

```
def MFCCExtractor(FeatureExtractor):

    def __init__(self, n_mfccs: int=13):
        # assign attributes here
    
    def compute_features(self, waveform: np.ndarray):
        # compute mfccs here

        return my_mfccs
```

From there on, just exchange the old extractor class used in ```main.py``` with your newly built one. <br>

Adding your own ```DataProcessor``` is just as easy. Just add a new class to ```data_processor.py``` inheriting from the ```DataProcessor``` class. The code could look like this:

```
def MyProcessor(DataProcessor):

    def __init__(self, # your arguments):
        # assign attributes here if needed
    
    def process_data(self, data: np.ndarray):
        # processing step 1
        # processing step 2
        # ...
        # processing step 3
        return data
```
## 5. Licence

MIT License

Copyright (c) 2022 Max Hilsdorf

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE