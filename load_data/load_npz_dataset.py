import os
import numpy as np

def main():
    # ---------------------------------------------------------------------
    # 1) Load dataset
    # ---------------------------------------------------------------------
    input_dir = "/Users/vittorio/Desktop/GSoC_FFD_PC/output_directory"
    filename  = "training_poisson_four_channels.npz"
    full_path = os.path.join(input_dir, filename)

    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
        return

    data = np.load(full_path)
    inputs = data["inputs"]   # shape [num_samples, Nx, Ny, Nz, 4]
    targets = data["targets"] # shape [num_samples, Nx, Ny, Nz, 1]

    """
    Suppose 'inputs' contains:
      - inputs[..., 0] = u velocity
      - inputs[..., 1] = v velocity
      - inputs[..., 2] = w velocity
      - inputs[..., 3] = boundary code (0,1,2, etc.)
    """

    print(f"Loaded dataset from {full_path}")
    print(f"  inputs shape:  {inputs.shape}")
    print(f"  targets shape: {targets.shape}")

    num_samples, Nx, Ny, Nz, num_inp_channels = inputs.shape
    if num_inp_channels < 4:
        print("Error: we need 4 channels: (u, v, w, boundary_code).")
        return

    # ---------------------------------------------------------------------
    # 2) Extract velocity components and compute divergence
    # ---------------------------------------------------------------------
    # shape: [num_samples, Nx, Ny, Nz]
    u_vals = inputs[..., 0]
    v_vals = inputs[..., 1]
    w_vals = inputs[..., 2]

    du_dx = np.gradient(u_vals, axis=1)  # partial derivative wrt x
    dv_dy = np.gradient(v_vals, axis=2)  # wrt y
    dw_dz = np.gradient(w_vals, axis=3)  # wrt z

    div_vals = du_dx + dv_dy + dw_dz   # shape: [num_samples, Nx, Ny, Nz]

    # Expand to [num_samples, Nx, Ny, Nz, 1]
    div_vals_5d = np.expand_dims(div_vals, axis=-1)

    print(f"Divergence computed. div_vals_5d shape = {div_vals_5d.shape}")

    # ---------------------------------------------------------------------
    # 3) Create one-hot boundary channels
    # ---------------------------------------------------------------------
    # boundary codes shape: [num_samples, Nx, Ny, Nz]
    bc_vals = inputs[..., 3]

    # We assume bc=0 => free, bc=1 => wall, bc=2 => inlet.
    # We'll create 3 separate channels (0 or 1) for each category.
    bc_free  = (bc_vals == 0).astype(np.float32)  # shape [num_samples, Nx, Ny, Nz]
    bc_wall  = (bc_vals == 1).astype(np.float32)
    bc_inlet = (bc_vals == 2).astype(np.float32)

    # Expand each to 5D [num_samples, Nx, Ny, Nz, 1]
    bc_free_5d  = np.expand_dims(bc_free,  axis=-1)
    bc_wall_5d  = np.expand_dims(bc_wall,  axis=-1)
    bc_inlet_5d = np.expand_dims(bc_inlet, axis=-1)

    print(f"bc_free_5d shape:  {bc_free_5d.shape}")
    print(f"bc_wall_5d shape:  {bc_wall_5d.shape}")
    print(f"bc_inlet_5d shape: {bc_inlet_5d.shape}")

    # ---------------------------------------------------------------------
    # 4) Concatenate into final shape => Nx,Ny,Nz,4
    # ---------------------------------------------------------------------
    # For each sample, we want 4 channels:
    #   0 => divergence
    #   1 => bc_free
    #   2 => bc_wall
    #   3 => bc_inlet
    #
    # new_data: [num_samples, Nx, Ny, Nz, 4]
    new_data = np.concatenate([
        div_vals_5d,  # shape [...,1]
        bc_free_5d,
        bc_wall_5d,
        bc_inlet_5d
    ], axis=-1)

    print(f"new_data shape = {new_data.shape}")

    # (Optionally) keep 'targets' the same or rename them:
    # If you only changed the input channels, your targets remain the same.

    # ---------------------------------------------------------------------
    # 5) Save the final data
    # ---------------------------------------------------------------------
    # We'll call it "training_divergence_onehot.npz".
    # It will contain:
    #   "inputs" => new_data
    #   "targets" => the same old targets
    output_dir = input_dir
    new_filename = "training_divergence_onehot.npz"
    new_full_path = os.path.join(output_dir, new_filename)

    np.savez(new_full_path, inputs=new_data, targets=targets)
    print(f"Saved final dataset with shape [num_samples, {Nx}, {Ny}, {Nz}, 4] to {new_full_path}")

if __name__ == "__main__":
    main()
