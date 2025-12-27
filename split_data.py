import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(args):
    """
    Splits the dataset into Train (70%), Val (10%), and Test (20%) sets
    enforcing Patient-Level Non-Overlapping constraint.
    
    Ref: Manuscript Section III-B "Data Partitioning"
    Seed: 42 (Fixed)
    """
    print(f"Reading full dataset metadata from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # 1. Check for 'Patient ID' column (Standard in ChestX-ray14)
    # If using PadChest or custom dataset, ensure a 'PatientID' or similar column exists.
    if 'Patient ID' not in df.columns:
        # Fallback for datasets without explicit Patient ID (assume strict random split)
        # But for ChestX-ray14, this column is standard.
        print("Warning: 'Patient ID' column not found. Using 'Image Index' as unique ID (Not recommended for clinical data).")
        patient_ids = df['Image Index'].unique()
    else:
        patient_ids = df['Patient ID'].unique()
    
    print(f"Total Unique Patients: {len(patient_ids)}")
    
    # 2. First Split: Separate Test set (20%)
    # Train+Val = 80%
    train_val_ids, test_ids = train_test_split(
        patient_ids, 
        test_size=0.20, 
        random_state=42, # Fixed Seed as per paper
        shuffle=True
    )
    
    # 3. Second Split: Separate Train (70% of total) and Val (10% of total)
    # The remaining 80% needs to be split into 7:1 ratio (0.8 * 0.125 = 0.1)
    # 10% of total is 12.5% of the remaining 80%
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.125, # 0.125 * 0.8 = 0.1 (10% of total)
        random_state=42,
        shuffle=True
    )
    
    print(f"Split Ratios - Train: {len(train_ids)} patients, Val: {len(val_ids)} patients, Test: {len(test_ids)} patients")
    
    # 4. Map back to Images
    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]
    test_df = df[df['Patient ID'].isin(test_ids)]
    
    print(f"Image Counts - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 5. Save to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_save_path = os.path.join(args.output_dir, 'train_labels.csv')
    val_save_path = os.path.join(args.output_dir, 'val_labels.csv')
    test_save_path = os.path.join(args.output_dir, 'test_labels.csv')
    
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    
    print(f"Saved split files to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Dataset with Patient-Level Isolation")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the original Data_Entry_2017.csv")
    parser.add_argument("--output_dir", type=str, default="./data_splits", help="Directory to save train/val/test csvs")
    
    args = parser.parse_args()
    split_dataset(args)