import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.mqfusionnet import MQFusionNet
from data.dataset import ChestXRayDataset, get_transforms
from utils.metrics import MetricCalculator

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")
    
    # Load Data
    test_dataset = ChestXRayDataset(args.data_dir, args.test_csv, transform=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load Model
    model = MQFusionNet(num_classes=14).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print("Error: Weights file not found!")
        return

    model.eval()
    metrics = MetricCalculator()
    
    print(">>> Starting Inference...")
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            metrics.update(labels, logits)
            
            if i % 10 == 0:
                print(f"Processed batch {i}/{len(test_loader)}")
    
    results = metrics.compute()
    print("="*30)
    print(f"Final Test Results:")
    print(f"Aggregate AUC: {results['AUC']:.4f}")
    print(f"MAE:           {results['MAE']:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    test(args)