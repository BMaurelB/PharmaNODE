import pandas as pd
import torch
import numpy as np
# Load the CSV file into a pandas DataFrame
try:
    df = pd.read_csv('/Users/benjaminmaurel/tacro_mapbayest_auc_20250826.csv')

    # --- Data Preparation ---
    # Drop rows with any missing values to ensure clean tensors
    df.dropna(inplace=True)

    # Define which columns are features (inputs) and which are labels (outputs)
    feature_columns = ['ST', 'amt', 'ii', 'CYP']
    label_columns = ['AUC_ii', 'auc_ipred'] 

    # Create separate DataFrames for features and labels
    features_df = df[feature_columns]
    labels_df = df[label_columns]

    # --- Convert to PyTorch Tensors ---
    # Convert the DataFrames to PyTorch tensors with a 32-bit float type
    features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_df.values, dtype=torch.float32)

    # --- Print results to verify ---
    print("--- PyTorch Tensors ---")
    print(f"Features Tensor Shape: {features_tensor.shape}")
    print("First 5 rows of features tensor:\n", features_tensor[:5])
    print(f"\nLabels Tensor Shape: {labels_tensor.shape}")
    print("First 5 rows of labels tensor:\n", labels_tensor[:5])
    ref = np.array(labels_df['AUC_ii'].tolist())
    pred = np.array(labels_df['auc_ipred'].tolist())
    rmse_be_overall = np.sqrt(np.mean(((ref - pred) / ref)**2))
    import pdb; pdb.set_trace()
except FileNotFoundError:
    print("Error: The file 'tacro_mapbayest_auc_20250826.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")