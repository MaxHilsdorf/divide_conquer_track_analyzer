from concurrent.futures import process
#import librosa
#from librosa.display import specshow
import matplotlib.pyplot as plt
from pydub import AudioSegment
from feature_extractor import FeatureExtractor
from data_processor import DataProcessor
from tensorflow import keras
import numpy as np

## HELPER FUNCTIONS ##

def do_padding(data: np.ndarray, pad_val: float, target_shape: tuple, pad_back=True):
    """Pads an array with zeros until a target shape is reached"""
    
    paddings = ()
    #print("Spec shape current:", data.shape)
    #print("Desired spec shape:", target_shape)
    for i, shape in enumerate(data.shape):
        
        if shape < target_shape[i]:
            #print(f"dim {i} is padded")
            
            diff = target_shape[i] - shape
            
            if pad_back:
                paddings += ((0,diff),)
            else:
                paddings += ((diff,0),)
                
        else:
            paddings += ((0,0),)
                
    return np.pad(data, paddings, mode="constant", constant_values=pad_val)


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
        
        #audio, _ = librosa.load(self.mp3_path, sr=sr)
        #audio_bytes = audio.tobytes()
        #self.audio_segment = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=sr, channels=1)
        
        self.audio_segment = AudioSegment.from_file(self.mp3_path, "mp3", frame_rate=sr)
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
        for i in range(self.slice_duration-self.overlap, self.duration-self.slice_duration+1, self.slice_duration-self.overlap): # add the rest
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
                self.waveform = np.array(self.audio_segment.get_array_of_samples(), dtype="float16")
                self.feature = feature_extractor.compute_feature(waveform=self.waveform)

class TrackAnalyzer:
    """Can be used to make predictions for all sub segments of a track with keras models"""

    def __init__(self, model: keras.Model, track: Track, processor: DataProcessor, feature_shape: tuple):

        self.model = model
        self.feature_shape = feature_shape
        self.track = track
        self.processor = processor
        
        self.features = self.get_feature_matrix()
        #print("Track name:", self.track.name)
        #print("Given Features shape:", self.feature_shape)
        #print("True Features shape:", self.features.shape)
        #print("Features min max:", self.features.min(), self.features.max())
        #print(self.features[-1,:,:,0])
        #specshow(self.features[0,:,:,0].T)
        #plt.show()

    def get_feature_matrix(self):

        # Aggregate features
        features = np.zeros((len(self.track.sub_segments), self.feature_shape[0], self.feature_shape[1]))

        for i, subseg in enumerate(self.track.sub_segments):
            
            feature = subseg.feature
    
            # Apply zero-padding if shape does not fit
            if feature.shape != features.shape[1:]:
                feature = do_padding(data=feature, pad_val=-80, target_shape=self.feature_shape, pad_back=True)
            
            features[i,:,:] = feature
        #print("Feature shape before processing:", features.shape)

        # Process features
        features = self.processor.process_data(features)
        #print("Feature shape after processing:", features.shape)

        return features
        

    def predict_sub_segments(self):

        return self.model.predict(self.features)
