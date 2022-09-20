
from concurrent.futures import process
from pydub import AudioSegment
from feature_extractor import FeatureExtractor
from data_processor import DataProcessor
from tensorflow import keras
import numpy as np

## HELPER FUNCTIONS ##

def zero_pad(data: np.ndarray, target_shape: tuple, pad_back=True):
    """Pads an array with zeros until a target shape is reached"""
    
    paddings = ()
    for i, shape in enumerate(data.shape):
        
        if shape < target_shape[i]:
            
            diff = target_shape[i] - shape
            
            if pad_back:
                paddings += ((0,diff),)
            else:
                paddings += ((diff,0),)
                
        else:
            paddings += ((0,0),)
                
    return np.pad(data, paddings)


## CLASSES ##

class Track:
    """Represents a track with all its individual segments"""

    def __init__(self, mp3_path: str, feature_extractor: FeatureExtractor, 
                 sr: int=22050, name: str=None, slice_duration: int=10, overlap: int=0):

        # Global information
        self.mp3_path = mp3_path
        if name:
            self.name = name
        else:
            self.name = ".".join(self.mp3_path.split("/")[-1].split(".")[:-1])
        self.audio_segment = AudioSegment.from_mp3(self.mp3_path)
        self.duration = int(self.audio_segment.duration_seconds) # round down
        
        # Snippet processing information
        self.sr = sr
        self.overlap = overlap
        self.slice_duration = slice_duration
        self.feature_extractor = feature_extractor
        
        # Get sub segments
        self.sub_segments = self.build_segments()

    def build_segments(self):

        # Compute timestamps
        slice_timestamps =[(0, self.slice_duration*1000)] # add first sample
        for i in range(self.slice_duration-self.overlap, self.duration-self.overlap, self.slice_duration-self.overlap): # add the rest
            slice_timestamps.append((i*1000, ((i+self.slice_duration)*1000)))

        # Build Segments
        sub_segments = []
        for ts in slice_timestamps:
            sub_segments.append(TrackSegment(track=self, timestamp=ts, feature_extractor=self.feature_extractor))
        return sub_segments

class TrackSegment:
    """Represents a single segment within a track"""

    def __init__(self, track: Track, timestamp: tuple, feature_extractor: FeatureExtractor):

                self.timestamp = timestamp
                self.audio_segment = track.audio_segment[timestamp[0]:timestamp[1]]
                self.waveform = np.array(self.audio_segment.get_array_of_samples(), dtype="float32")
                self.feature = feature_extractor.compute_feature(waveform=self.waveform)

class TrackAnalyzer:
    """Can be used to make predictions for all sub segments of a track with keras models"""

    def __init__(self, model: keras.Model, track: Track, processor: DataProcessor, feature_shape: tuple):

        self.model = model
        self.feature_shape = feature_shape
        self.track = track
        self.processor = processor
        
        self.features = self.get_feature_matrix()

    def get_feature_matrix(self):

        # Aggregate features
        features = np.zeros((len(self.track.sub_segments), self.feature_shape[0], self.feature_shape[1]))

        for i, subseg in enumerate(self.track.sub_segments):
            
            feature = subseg.feature
    
            # Apply zero-padding if shape does not fit
            if feature.shape != features.shape[1:]:
                feature = zero_pad(data=feature, target_shape=features.shape[1:], pad_back=True)
            
            features[i,:,:] = feature

        # Process features
        features = self.processor.process_data(features)

        return features
        

    def predict_sub_segments(self):

        return self.model.predict(self.features)
