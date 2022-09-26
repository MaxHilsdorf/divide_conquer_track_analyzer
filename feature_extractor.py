from functools import lru_cache
from pydub import AudioSegment
import librosa
import numpy as np

class FeatureExtractor:
    """Mother class and blueprint for custom feature extractors"""
    
    def __init__(self, data_type="float32"):
        pass
    
    def compute_feature(self, waveform: np.ndarray):
        pass
    
class MelspecExtractor(FeatureExtractor):
    """Extractor for mel spectrograms using librosa"""
    
    def __init__(self, sr: int=22050, n_fft: int=2048,
                 hop_length: int=512, n_mels: int=100, data_type="float16"):
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.data_type = data_type
    
    def compute_feature(self, waveform: np.ndarray):
        
        melspec = librosa.feature.melspectrogram(y=waveform, sr=self.sr, n_fft=self.n_fft,
                                                    hop_length=self.hop_length, n_mels=self.n_mels)
        melspec = librosa.power_to_db(melspec, ref=np.max)
        
        return melspec.astype(self.data_type)
        
        
        
