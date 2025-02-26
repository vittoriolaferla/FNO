import os
import numpy as np

def main():
    # ---------------------------------------------------------------------
    # 1) Set up the parent directory containing subfolders
    # ---------------------------------------------------------------------
    parent_dir = "/Users/vittorio/Desktop/FNO/all_output_less"
    filename   = "training_poisson_four_channels.npz"

    # ---------------------------------------------------------------------
    # 2) Gather subfolders
    # ---------------------------------------------------------------------
    # We only take paths that are directories:
    subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

    # Containers for collecting processed data across subfolders:
    # We will concatenate them at the end along the sample dimension (axis=0).
    all_new_data = []
    all_targets  = []

    # ---------------------------------------------------------------------
    # 3) Loop over subfolders, load, process, and collect data
    # ---------------------------------------------------------------------
    for subfolder in subfolders:
        full_path = os.path.join(subfolder, filename)

        if not os.path.isfile(full_path):
            print(f"Warning: File not found in {subfolder}: {filename}")
            continue

        # Load file
        data = np.load(full_path)
        inputs  = data["inputs"]   # shape [num_samples, Nx, Ny, Nz, 4]
        targets = data["targets"]  # shape [num_samples, Nx, Ny, Nz, 1]

        print(f"Loaded dataset from {full_path}")
        print(f"  inputs shape:  {inputs.shape}")
        print(f"  targets shape: {targets.shape}")

        num_samples, Nx, Ny, Nz, num_inp_channels = inputs.shape
        if num_inp_channels < 4:
            print("Error: we need at least 4 channels: (u, v, w, boundary_code). Skipping folder.")
            continue

        # ---------------------------------------------------------------------
        # 3a) Extract velocity components and compute divergence
        # ---------------------------------------------------------------------
        # shape: [num_samples, Nx, Ny, Nz]
        u_vals = inputs[..., 0]
        v_vals = inputs[..., 1]
        w_vals = inputs[..., 2]

        du_dx = np.gradient(u_vals, axis=1)  # partial derivative wrt x
        dv_dy = np.gradient(v_vals, axis=2)  # wrt y
        dw_dz = np.gradient(w_vals, axis=3)  # wrt z

        div_vals = du_dx + dv_dy + dw_dz     # shape: [num_samples, Nx, Ny, Nz]
        # Expand to [num_samples, Nx, Ny, Nz, 1]
        div_vals_5d = np.expand_dims(div_vals, axis=-1)

        # ---------------------------------------------------------------------
        # 3b) Create one-hot boundary channels
        # ---------------------------------------------------------------------
        bc_vals = inputs[..., 3]  # shape: [num_samples, Nx, Ny, Nz]

        bc_free  = (bc_vals == 0).astype(np.float32)
        bc_wall  = (bc_vals == 1).astype(np.float32)
        bc_inlet = (bc_vals == 2).astype(np.float32)

        bc_free_5d  = np.expand_dims(bc_free,  axis=-1)
        bc_wall_5d  = np.expand_dims(bc_wall,  axis=-1)
        bc_inlet_5d = np.expand_dims(bc_inlet, axis=-1)

        # ---------------------------------------------------------------------
        # 3c) Concatenate into final shape => Nx, Ny, Nz, 4
        # ---------------------------------------------------------------------
        # new_data shape: [num_samples, Nx, Ny, Nz, 4]
        new_data = np.concatenate([
            div_vals_5d,
            bc_free_5d,
            bc_wall_5d,
            bc_inlet_5d
        ], axis=-1)

        print(f"new_data shape = {new_data.shape}")

        # Collect them to concatenate later
        all_new_data.append(new_data)
        all_targets.append(targets)

    # ---------------------------------------------------------------------
    # 4) Concatenate all data across subfolders
    # ---------------------------------------------------------------------
    if len(all_new_data) == 0:
        print("No processed data found. Exiting.")
        return

    final_inputs  = np.concatenate(all_new_data, axis=0)  # shape [sum_samples, Nx, Ny, Nz, 4]
    final_targets = np.concatenate(all_targets,  axis=0)  # shape [sum_samples, Nx, Ny, Nz, 1]

    print(f"final_inputs shape  = {final_inputs.shape}")
    print(f"final_targets shape = {final_targets.shape}")

    # ---------------------------------------------------------------------
    # 5) Save the final dataset
    # ---------------------------------------------------------------------
    # We'll call it "training_divergence_onehot_combined.npz"
    output_path = os.path.join(parent_dir, "training_divergence_onehot_combined.npz")
    np.savez(output_path, inputs=final_inputs, targets=final_targets)
    print(f"Saved combined dataset to: {output_path}")


if __name__ == "__main__":
    main()
