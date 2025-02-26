import os
import numpy as np

def main():
    # ---------------------------------------------------------------------
    # 1) Specify input file and output names
    # ---------------------------------------------------------------------
    input_file = "/Users/vittorio/Desktop/FNO/all_output_less/training_divergence_onehot_combined.npz"
    train_file = "/Users/vittorio/Desktop/FNO/all_output_less/training_set.npz"
    test_file  = "/Users/vittorio/Desktop/FNO/all_output_less/test_set.npz"

    # Make sure the input file exists
    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        return

    # ---------------------------------------------------------------------
    # 2) Load the combined dataset
    # ---------------------------------------------------------------------
    data = np.load(input_file)
    inputs  = data["inputs"]   # shape: [N, Nx, Ny, Nz, 4]
    targets = data["targets"]  # shape: [N, Nx, Ny, Nz, 1]

    total_samples = inputs.shape[0]
    print(f"Loaded '{input_file}' with {total_samples} total samples.")

    # ---------------------------------------------------------------------
    # 3) Create train/test split
    # ---------------------------------------------------------------------
    # Decide on a train/test ratio
    train_ratio = 0.8
    train_count = int(train_ratio * total_samples)

    # Shuffle indices to randomize
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Partition into train and test indices
    train_indices = indices[:train_count]
    test_indices  = indices[train_count:]

    # Split the data
    train_inputs  = inputs[train_indices]
    train_targets = targets[train_indices]
    test_inputs   = inputs[test_indices]
    test_targets  = targets[test_indices]

    print(f"Training set: {train_inputs.shape[0]} samples")
    print(f"Test set:     {test_inputs.shape[0]} samples")

    # ---------------------------------------------------------------------
    # 4) Save the resulting datasets
    # ---------------------------------------------------------------------
    np.savez(train_file, inputs=train_inputs, targets=train_targets)
    np.savez(test_file,  inputs=test_inputs,  targets=test_targets)
    print(f"Saved training set to: '{train_file}'")
    print(f"Saved test set to:     '{test_file}'")

if __name__ == "__main__":
    main()
