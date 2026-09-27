import os
import torch
import torch.nn as nn
from models.deepshield_model import DeepShieldEngine
from .dataset_loader import get_dataloader

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        spatial = batch['spatial'].to(device)
        freq = batch['freq'].to(device)
        landmarks = batch['landmarks'].to(device)
        global_img = batch['global_img'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Automatic Mixed Precision (AMP) for Colab/Kaggle GPU speedup
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(spatial, freq, landmarks, global_img)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def run_cross_dataset_scenarios():
    scenarios = {
        "Scenario_1": {"train": ["FF++", "DFDC"], "test": ["UADFV"]},
        "Scenario_2": {"train": ["FF++", "UADFV"], "test": ["DFDC"]},
        "Scenario_3": {"train": ["DFDC", "UADFV"], "test": ["FF++"]},
        "Scenario_4": {"train": ["FF++", "DFDC", "UADFV"], "test": ["HOLDOUT"]}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Training Matrix on Device: {device}")

    for sc_name, cfg in scenarios.items():
        print(f"\n==================== {sc_name} ====================")
        print(f"Train Sets: {cfg['train']} ---> Test Sets: {cfg['test']}")
        
        model = DeepShieldEngine(embed_dim=256).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        dataloader = get_dataloader(batch_size=2, num_samples=10, seq_len=5)
        
        loss = train_one_epoch(model, dataloader, optimizer, criterion, scaler, device)
        print(f"[{sc_name}] Initial Verification Epoch Loss: {loss:.4f}")

if __name__ == "__main__":
    run_cross_dataset_scenarios()