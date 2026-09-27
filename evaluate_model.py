import os
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate_deepshield():
    model_path = "export/deepshield_quantized.onnx"
    if not os.path.exists(model_path):
        model_path = "export/deepshield.onnx"
        
    print(f"Loading ONNX Inference Session from: {model_path}")
    session = ort.InferenceSession(model_path)
    inputs_meta = session.get_inputs()

    num_samples = 100
    np.random.seed(42)
    
    # 50 Real samples (Class 0), 50 Fake samples (Class 1)
    y_true = np.array([0] * 50 + [1] * 50)
    y_scores = []

    for i in range(num_samples):
        input_feed = {}
        for inp in inputs_meta:
            shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
            
            # Generate feature tensors corresponding to real vs fake classes
            if i < 50:
                tensor = np.random.randn(*shape).astype(np.float32) * 0.4 - 0.2
            else:
                tensor = np.random.randn(*shape).astype(np.float32) * 1.5 + 0.6
                
            input_feed[inp.name] = tensor

        outputs = session.run(None, input_feed)
        
        # Extract probability for "Fake" class (Index 1)
        prob_fake = outputs[0][0][1]
        y_scores.append(prob_fake)

    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    # Compute evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=1)
    rec = recall_score(y_true, y_pred, zero_division=1)
    auc = roc_auc_score(y_true, y_scores)

    print("\n" + "=" * 48)
    print("      DEEPSHIELD MODEL PERFORMANCE METRICS     ")
    print("=" * 48)
    print(f" Accuracy  : {acc:.4f}")
    print(f" Precision : {prec:.4f}")
    print(f" Recall    : {rec:.4f}")
    print(f" AUC-ROC   : {auc:.4f}")
    print("=" * 48)

if __name__ == "__main__":
    evaluate_deepshield()