import numpy as np

## Normalization functions ##

def minmax_normalization(data: np.ndarray, delta: int=0):
    """Normalizes the data to the interval [0+delta, 1+delta]"""
    
    return (data - data.min()) / (data.max() - data.min()) + delta

## Processor classes ##

class DataProcessor():
    """Mother class and blueprint for custom data processors"""
    def __init__(self):
        pass

    def process_data(self, data:np.ndarray):
        pass
    
class MelspecCNNProcessor(DataProcessor):
    """
    Processor for mel spectrograms to be processed by CNNs
    Implements
    -   non-zero min-max normalization
    -   dimension expansion for CNN "color" channel
    """
    
    def __init__(self, delta=0.001):
        self.delta = delta
    
    def process_data(self, data:np.ndarray):
        
        data = minmax_normalization(data, self.delta)
        data = np.expand_dims(data, 3)
        
        return data
    
    
class MelspecCRNNProcessor(DataProcessor):
    """
    Processor for mel spectrograms to be processed by CRNNs
    Implements
    -   non-zero min-max normalization
    -   dimension expansion for CNN "color" channel
    -   ax swap to put time dimension in second place
    """
    
    def __init__(self, delta=0.001):
        self.delta = delta
    
    def process_data(self, data:np.ndarray):
        
        data = minmax_normalization(data, delta=self.delta)
        data = np.expand_dims(data, 3)
        data = np.swapaxes(data, 1, 2)
        
        return data