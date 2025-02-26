import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from timeit import default_timer

from tqdm import tqdm  # <-- Import tqdm

from dataset import DivBoundaryOneHotDataset, UnitGaussianNormalizer
from model import FNO3d
from model import CombinedLoss


def main():
    # --------------------------
    # A) Configuration
    # --------------------------
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Update these paths and filenames to match your setup
    data_dir = "/Users/vittorio/Desktop/FNO/output_directory"
    train_file = "training_set.npz"
    test_file  = "test_set.npz"

    train_path = os.path.join(data_dir, train_file)
    test_path  = os.path.join(data_dir, test_file)

    if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
        print("Could not find training and/or test files.")
        print(f"  Train file path: {train_path}")
        print(f"  Test file path:  {test_path}")
        return

    # --------------------------
    # B) Load training and test data
    # --------------------------
    np_train_data = np.load(train_path)
    x_train_np = np_train_data["inputs"]   # shape: [N_train, Nx, Ny, Nz, 4]
    y_train_np = np_train_data["targets"]  # shape: [N_train, Nx, Ny, Nz, 1]

    np_test_data = np.load(test_path)
    x_test_np = np_test_data["inputs"]     # shape: [N_test, Nx, Ny, Nz, 4]
    y_test_np = np_test_data["targets"]    # shape: [N_test, Nx, Ny, Nz, 1]

    print(f"Training data loaded from: {train_path}")
    print(f"  x_train shape: {x_train_np.shape}")
    print(f"  y_train shape: {y_train_np.shape}")

    print(f"Test data loaded from: {test_path}")
    print(f"  x_test shape:  {x_test_np.shape}")
    print(f"  y_test shape:  {y_test_np.shape}")

    # Convert numpy arrays to torch tensors
    x_train = torch.from_numpy(x_train_np).float()
    y_train = torch.from_numpy(y_train_np).float()
    x_test  = torch.from_numpy(x_test_np).float()
    y_test  = torch.from_numpy(y_test_np).float()

    N_train = x_train.shape[0]
    N_test  = x_test.shape[0]

    # --------------------------
    # C) Normalization
    # --------------------------
    # Here, we normalize only channel 0 (the divergence) for x,
    # and channel 0 (the pressure) for y.
    x_normalizer = UnitGaussianNormalizer(x_train, channel_ids_to_normalize=(0,))
    y_normalizer = UnitGaussianNormalizer(y_train, channel_ids_to_normalize=(0,))

    x_train = x_normalizer.encode(x_train)
    y_train = y_normalizer.encode(y_train)
    x_test  = x_normalizer.encode(x_test)
    y_test  = y_normalizer.encode(y_test)

    # --------------------------
    # D) Datasets / Loaders
    # --------------------------
    train_dataset = DivBoundaryOneHotDataset(x_train, y_train)
    test_dataset  = DivBoundaryOneHotDataset(x_test, y_test)

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # --------------------------
    # E) FNO Model
    # --------------------------
    in_channels  = 4   # [div, free, wall, inlet]
    out_channels = 1   # pressure
    modes = 8
    width = 24
    depth = 4

    model = FNO3d(
        modes1=modes,
        modes2=modes,
        modes3=modes,
        width=width,
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth
    ).to(device)

    # --------------------------
    # F) Training Setup
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = CombinedLoss(
        data_loss_weight=1.0,
        pde_loss_weight=0.1,
        p=2,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer
    )
    epochs = 5

    print(f"Training FNO3D with PDE loss, using {N_train} train samples and {N_test} test samples.")

    # --------------------------
    # G) Training Loop
    # --------------------------
    for ep in range(epochs):
        t0 = default_timer()

        # ---- Training Phase ----
        model.train()
        total_loss = 0.0

        # Wrap train_loader with tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} [Training]", leave=False)
        for x_batch, y_batch in train_pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)  # shape [b, Nx, Ny, Nz, 1]

            # Pass x_batch as well for PDE loss
            loss_val = loss_fn(pred, y_batch, x_batch)
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item() * x_batch.size(0)

            # Optionally, update tqdm postfix with current batch loss
            train_pbar.set_postfix({"batch_loss": loss_val.item()})

        train_loss = total_loss / len(train_dataset)

        # ---- Validation (Test) Phase ----
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            # Wrap test_loader with tqdm for a progress bar
            test_pbar = tqdm(test_loader, desc=f"Epoch {ep+1}/{epochs} [Testing]", leave=False)
            for x_batch, y_batch in test_pbar:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)

                loss_val = loss_fn(pred, y_batch, x_batch)
                test_loss += loss_val.item() * x_batch.size(0)

                # Optionally, update tqdm postfix with current batch loss
                test_pbar.set_postfix({"batch_loss": loss_val.item()})

        test_loss /= len(test_dataset)
        t1 = default_timer()

        print(f"Epoch {ep+1}/{epochs} | "
              f"TrainLoss: {train_loss:.6f}, TestLoss: {test_loss:.6f}, "
              f"Time: {t1 - t0:.2f}s")

        # Save the model every 2 epochs
        if (ep + 1) % 2 == 0:
            checkpoint_path = f"fno_div_onehotBC_to_pressure_epoch_{ep+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # --------------------------
    # H) Save Final Model
    # --------------------------
    final_model_path = "fno_div_onehotBC_to_pressure.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to '{final_model_path}'.")


if __name__ == "__main__":
    main()
