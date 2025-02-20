import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from timeit import default_timer

from dataset import DivBoundaryOneHotDataset, UnitGaussianNormalizer
from model import FNO3d
from model import  CombinedLoss 

def main():
    # --------------------------
    # A) Configuration
    # --------------------------
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "/Users/vittorio/Desktop/FNO/output_directory"
    data_file = "training_divergence_onehot.npz"
    full_path = os.path.join(data_dir, data_file)

    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
        return

    # --------------------------
    # B) Load data
    # --------------------------
    np_data = np.load(full_path)
    x_np = np_data["inputs"]   # shape [N, 48,48,48, 4]
    y_np = np_data["targets"]  # shape [N, 48,48,48, 1]

    print(f"Loaded dataset from: {full_path}")
    print(f"  inputs shape:  {x_np.shape}")
    print(f"  targets shape: {y_np.shape}")

    x_all = torch.from_numpy(x_np).float()
    y_all = torch.from_numpy(y_np).float()

    N = x_all.shape[0]
    N_train = int(0.8 * N)
    N_test  = N - N_train

    x_train = x_all[:N_train]
    y_train = y_all[:N_train]
    x_test  = x_all[N_train:]
    y_test  = y_all[N_train:]

    # --------------------------
    # C) Normalization
    # --------------------------
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

    model = FNO3d(modes, modes, modes, width, in_channels, out_channels, depth).to(device)

    # --------------------------
    # F) Training Setup
    # --------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # We instantiate the combined loss with PDE weight = 0.1 (example)
    loss_fn = CombinedLoss(
        data_loss_weight=1.0, 
        pde_loss_weight=0.1, 
        p=2,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer
    )
    epochs = 5

    print(f"Training FNO3D with PDE loss, using {N_train} train and {N_test} test samples.")

    # --------------------------
    # G) Training Loop
    # --------------------------
    for ep in range(epochs):
        t0 = default_timer()
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)  # shape [b, 48,48,48,1]

            # *** Notice we pass x_batch as well for PDE loss
            loss_val = loss_fn(pred, y_batch, x_batch)  
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item() * x_batch.size(0)

        train_loss = total_loss / len(train_dataset)

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss_val = loss_fn(pred, y_batch, x_batch)
                test_loss += loss_val.item() * x_batch.size(0)

        test_loss /= len(test_dataset)
        t1 = default_timer()
        print(f"Epoch {ep+1}/{epochs} | "
              f"TrainLoss: {train_loss:.6f}, TestLoss: {test_loss:.6f}, "
              f"Time: {t1 - t0:.2f}s")

    # --------------------------
    # H) Save Model
    # --------------------------
    torch.save(model.state_dict(), "fno_div_onehotBC_to_pressure.pt")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
