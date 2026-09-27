import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
from preprocessing.frequency_transform import FrequencyTransform
from preprocessing.landmark_graph import LandmarkGraphExtractor

class LocalInferencePipeline:
    def __init__(self, model_path="export/deepshield_quantized.onnx"):
        if not os.path.exists(model_path):
            model_path = "export/deepshield.onnx"
            
        print(f"Loading ONNX Inference Session: {model_path}")
        self.session = ort.InferenceSession(model_path)
        
        # Introspect model metadata to dynamically bind input names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"Detected ONNX Input Signature: {self.input_names}")

        self.freq_transform = FrequencyTransform()
        self.landmark_extractor = LandmarkGraphExtractor()

    def preprocess_video(self, video_path, num_frames=5):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_resized)
            
        cap.release()

        # If video is shorter than required frames, pad with zero frames
        while len(frames) < num_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        spatial_list = []
        freq_list = []
        landmark_list = []

        for frame in frames:
            # 1. Spatial Tensor [3, 224, 224] normalized
            spatial_tensor = frame.astype(np.float32) / 255.0
            spatial_tensor = np.transpose(spatial_tensor, (2, 0, 1))
            spatial_list.append(spatial_tensor)

            # 2. Frequency Spectrum Tensor [3, 224, 224]
            freq_tensor = self.freq_transform.compute_fft(frame)
            freq_list.append(freq_tensor)

            # 3. Landmark Coordinate Tensor [468, 3]
            landmarks = self.landmark_extractor.extract_landmarks(frame)
            if isinstance(landmarks, torch.Tensor):
                landmarks = landmarks.numpy()
            landmark_list.append(landmarks)

        # Batch & Sequence Stack: [Batch=1, Seq=5, ...]
        seq_spatial = np.expand_dims(np.array(spatial_list, dtype=np.float32), axis=0)
        seq_freq = np.expand_dims(np.array(freq_list, dtype=np.float32), axis=0)
        seq_landmarks = np.expand_dims(np.array(landmark_list, dtype=np.float32), axis=0)

        return seq_spatial, seq_freq, seq_landmarks, seq_spatial

    def predict(self, video_path):
        seq_spatial, seq_freq, seq_landmarks, seq_global = self.preprocess_video(video_path)
        
        # Map feature tensors dynamically to the model's expected input signature
        input_feed = {}
        for name in self.input_names:
            if 'spatial' in name.lower():
                input_feed[name] = seq_spatial
            elif 'freq' in name.lower():
                input_feed[name] = seq_freq
            elif 'landmark' in name.lower() or 'graph' in name.lower():
                input_feed[name] = seq_landmarks
            else:
                input_feed[name] = seq_global

        # Execute ONNX Inference
        outputs = self.session.run(None, input_feed)
        probabilities = outputs[0][0] # Softmax probabilities [Prob_Real, Prob_Fake]
        
        fake_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

        # Generate spatial activation heatmap visualization frame
        first_frame = (seq_spatial[0, 0].transpose(1, 2, 0) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(first_frame, 0.6, heatmap, 0.4, 0)

        return fake_probability, heatmap_overlay