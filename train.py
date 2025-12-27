import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mqfusionnet import MQFusionNet
from data.dataset import ChestXRayDataset, get_transforms
from utils.loss import ConsistencyLoss
from utils.metrics import MetricCalculator

def set_seed(seed=3407):
    # Fixed seed for reproducibility (Section III-B)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loaders
    train_dataset = ChestXRayDataset(args.data_dir, args.train_csv, transform=get_transforms('train'))
    val_dataset = ChestXRayDataset(args.data_dir, args.val_csv, transform=get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize Model
    model = MQFusionNet(num_classes=14).to(device)
    
    # Optimizer & Loss
    criterion = ConsistencyLoss(lambda_reg=args.lambda_reg)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1)
    
    best_auc = 0.0
    metrics = MetricCalculator()
    
    print(">>> Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Training Loop
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, alphas = model(imgs)
            
            loss, _, _ = criterion(logits, labels, alphas)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation Loop
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits, _ = model(imgs)
                metrics.update(labels, logits)
        
        val_results = metrics.compute()
        val_auc = val_results['AUC']
        val_mae = val_results['MAE']
        
        print(f"Epoch {epoch+1} Results | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val MAE: {val_mae:.4f}")
        
        scheduler.step(val_auc)
        
        # Save Best Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"New Best Model Saved! AUC: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train labels csv")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to val labels csv")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)