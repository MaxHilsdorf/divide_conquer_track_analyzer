import numpy as np
from tensorflow import keras

from feature_extractor import MelspecExtractor
from track_analyzer import Track, TrackAnalyzer
from data_processor import MelspecCRNNProcessor

MP3_PATH = "F:/music_datasets/gtzan_snippet/raw_mp3s/disco/disco.00000.mp3"
MODEL_PATH = "E:/OneDrive/uni_work/programming/data_science/gtzan/models/crnn_snippet/model.h5"
DATA_TYPE = "float16"

# Initialize extractor
extractor = MelspecExtractor(sr=22050, n_fft=20148, hop_length=512, n_mels=100, data_type=DATA_TYPE)

# Load track
track = Track(mp3_path=MP3_PATH, feature_extractor=extractor, sr=22050,
              name="example_track", slice_duration=3, overlap=1)

# Load model
model = keras.models.load_model(MODEL_PATH)
model_input_shape = np.array(model.input.shape)

# Initialize analyzer

analyzer = TrackAnalyzer(model=model, track=track, processor=MelspecCRNNProcessor(delta=0.01),
                         feature_shape = (model_input_shape[2], model_input_shape[1]))


print("Predicted values for all sub segments:")
print(analyzer.predict_sub_segments())
