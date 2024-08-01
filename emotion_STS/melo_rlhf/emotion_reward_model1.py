#!/home/ubuntu/OpenVoice/.openvoiceenv/bin/python
# Load model directly
# from transformers import AutoProcessor, Wav2Vec2ForSpeechClassification
from transformers import *
import torchaudio
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model
processor = AutoProcessor.from_pretrained("Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition")
model = Wav2Vec2ForSpeechClassification.from_pretrained("Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition")

def preprocess_audio(file_path):
    # Load and preprocess audio
    audio, sample_rate = torchaudio.load(file_path)
    audio = processor(audio.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values
    return audio

def get_emotion_embedding(audio):
    # Forward pass through the model to get emotion logits
    with torch.no_grad():
        outputs = model(audio)
        logits = outputs.logits
        # Optionally apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        # Here, we'll use logits directly as embeddings
        return logits.squeeze().numpy()

# Paths to your audio files
file_path_1 = "/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real/0011_000488_angry.wav"
file_path_2 = "/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real/0011_000852_happy.wav"

# Preprocess audio files
audio_1 = preprocess_audio(file_path_1)
audio_2 = preprocess_audio(file_path_2)

# Get emotion embeddings
embedding_1 = get_emotion_embedding(audio_1)
embedding_2 = get_emotion_embedding(audio_2)

# Calculate cosine similarity
similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]

print(f"Emotion similarity between the two audios: {similarity:.2f}")



# import torch
# import librosa
# from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
# from scipy.spatial.distance import cosine, euclidean

# # Define a function to load and preprocess audio files
# def load_and_preprocess(audio_path, feature_extractor):
#     speech, _ = librosa.load(audio_path, sr=16000, mono=True)
#     inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
#     return inputs

# # Load the pre-trained model and feature extractor
# model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")

# # Paths to your audio files
# audio_path1 = "/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real/0011_000488_angry.wav"
# audio_path2 = "/home/ubuntu/OpenVoice/emotion_STS/melo_emo/eval_input_audio/real/0011_000852_happy.wav"


# # Load and preprocess audio files
# inputs1 = load_and_preprocess(audio_path1, feature_extractor)
# inputs2 = load_and_preprocess(audio_path2, feature_extractor)

# # Extract embeddings
# with torch.no_grad():
#     logits1 = model(**inputs1).logits
#     logits2 = model(**inputs2).logits

# # Take the mean of the logits across the sequence length dimension to get a fixed-size embedding
# embedding1 = logits1.mean(dim=1).squeeze().numpy().flatten()
# embedding2 = logits2.mean(dim=1).squeeze().numpy().flatten()

# # Calculate similarities
# cosine_sim = 1 - cosine(embedding1, embedding2)
# euclidean_dist = euclidean(embedding1, embedding2)

# print(f"Cosine Similarity: {cosine_sim}")
# print(f"Euclidean Distance: {euclidean_dist}")

###############################################################