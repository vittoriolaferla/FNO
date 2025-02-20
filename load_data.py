#!/usr/bin/env python
import os
import numpy as np
import vtk
from vtk.util import numpy_support as vtk_np

def load_vtk_fields(filename, use_cell_data=False):
    """
    Reads a legacy VTK (rectilinear) file and returns a dict of arrays.
    
    If 'use_cell_data=True', we explicitly read the CELL_DATA instead of POINT_DATA.
    Otherwise, we default to reading POINT_DATA (with a fallback to CELL_DATA if empty).
    """
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()

    data_object = reader.GetOutput()

    if use_cell_data:
        # Force reading the CellData (since geometry is typically stored there).
        cell_data = data_object.GetCellData()
        n_arrays = cell_data.GetNumberOfArrays()
        if n_arrays == 0:
            raise ValueError(f"No arrays found in CELL_DATA of file {filename}")
        fields_dict = {}
        for i in range(n_arrays):
            array_vtk = cell_data.GetArray(i)
            if array_vtk is None:
                continue
            name = array_vtk.GetName()
            np_array = vtk_np.vtk_to_numpy(array_vtk)
            fields_dict[name] = np_array
        return fields_dict
    else:
        # Default: try POINT_DATA, fallback to CELL_DATA
        point_data = data_object.GetPointData()
        if point_data.GetNumberOfArrays() == 0:
            point_data = data_object.GetCellData()
        fields_dict = {}
        for i in range(point_data.GetNumberOfArrays()):
            array_vtk = point_data.GetArray(i)
            if array_vtk is None:
                continue
            name = array_vtk.GetName()
            np_array = vtk_np.vtk_to_numpy(array_vtk)
            fields_dict[name] = np_array
        return fields_dict

def reshape_field_3d(field_array, Nx, Ny, Nz, ncomp=1, order='F'):
    """
    Reshape a flat [Npoints, ncomp] array into a [Nx, Ny, Nz, ncomp] volume.
    The exact reshape depends on how the solver writes grid points (e.g., 'F' vs 'C').

    - field_array: shape [Npoints, ncomp] or [Npoints,] if ncomp=1
    - Nx, Ny, Nz: grid dimensions
    - ncomp: number of components (3 for velocity, 1 for scalar)
    - order: 'F' (column-major) or 'C' (row-major)
    """
    if field_array.ndim == 1:
        field_array = field_array[:, None]  # [Npoints, 1]

    Npoints = Nx * Ny * Nz
    if field_array.shape[0] != Npoints:
        raise ValueError(
            f"Field has {field_array.shape[0]} points, expected {Npoints} "
            f"for Nx={Nx}, Ny={Ny}, Nz={Nz}."
        )

    reshaped = field_array.reshape((Nx, Ny, Nz, ncomp), order=order)
    return reshaped

def build_training_dataset(
    directory,
    tsteps,
    Nx,
    Ny,
    Nz,
    bc_cropped,
    intermediate_pattern="roomVentilation_intermediate_{:d}.vtk",
    final_pattern="roomVentilation_{:d}.vtk",
):
    """
    Loops over a range of time steps, loads:
      - an 'intermediate' file (velocity_intermediate)
      - a 'final' file (pressure)
    Reshapes them into 4D arrays:
      - velocity: [Nx, Ny, Nz, 3]
      - pressure: [Nx, Ny, Nz, 1]
    Then appends the boundary code as the 4th channel => [Nx, Ny, Nz, 4] for velocity+bc.
    
    bc_cropped: array of shape (Nx, Ny, Nz) with boundary codes (e.g. 0,1,2,...)
    
    Returns:
      inputs_list: list of arrays [Nx, Ny, Nz, 4]
      targets_list: list of arrays [Nx, Ny, Nz, 1]
    """
    inputs_list = []
    targets_list = []

    # Expand bc_cropped to shape (Nx, Ny, Nz, 1) for concatenation
    bc_expanded = bc_cropped[..., None]  # => (Nx,Ny,Nz,1)

    for t in tsteps:
        # 1) "intermediate" file => velocity
        inter_file = os.path.join(directory, intermediate_pattern.format(t))
        if not os.path.isfile(inter_file):
            print(f"Warning: file not found {inter_file}, skipping timestep {t}.")
            continue
        inter_fields = load_vtk_fields(inter_file, use_cell_data=False)

        # 2) "final" file => pressure
        final_file = os.path.join(directory, final_pattern.format(t))
        if not os.path.isfile(final_file):
            print(f"Warning: file not found {final_file}, skipping timestep {t}.")
            continue
        final_fields = load_vtk_fields(final_file, use_cell_data=False)

        # 3) Extract velocity_intermediate => shape [Npoints, 3]
        if "velocity_intermediate" not in inter_fields:
            raise KeyError(
                f"'velocity_intermediate' not found in {inter_file}. "
                f"Fields available: {list(inter_fields.keys())}"
            )
        vel_intermediate = inter_fields["velocity_intermediate"]

        # 4) Extract pressure => shape [Npoints,]
        if "p_first" not in final_fields:
            raise KeyError(
                f"'pressure' not found in {final_file}. "
                f"Fields available: {list(final_fields.keys())}"
            )
        pressure = final_fields["p_first"]

        # 5) Reshape the fields
        vel_reshaped = reshape_field_3d(vel_intermediate, Nx, Ny, Nz, ncomp=3, order="F")
        press_reshaped = reshape_field_3d(pressure, Nx, Ny, Nz, ncomp=1, order="F")

        # 6) Concatenate boundary code => final input shape: [Nx, Ny, Nz, 4]
        vel_plus_bc = np.concatenate([vel_reshaped, bc_expanded], axis=-1)

        inputs_list.append(vel_plus_bc)     # [Nx,Ny,Nz,4]
        targets_list.append(press_reshaped)  # [Nx,Ny,Nz,1]

        print(
            f"Timestep {t}: velocity shape = {vel_reshaped.shape}, "
            f"bc shape = {bc_expanded.shape} => input shape = {vel_plus_bc.shape}, "
            f"pressure shape = {press_reshaped.shape}"
        )

    return inputs_list, targets_list

def main():
    # -----------------------------
    # 1) Setup parameters & paths
    # -----------------------------
    # Directory containing VTK files (adjust these paths as needed)
    directory = "/Users/vittorio/Desktop/FNO/simulations_output/0"
    tsteps = range(1, 100)  # time steps to process

    # Interior domain dimensions (velocity & pressure will be on a 48^3 grid)
    Nx, Ny, Nz = 48, 48, 48

    # Geometry file (contains boundary cell information)
    geometry_file = os.path.join(directory, "final", "roomVentilation_geometry.vtk")

    # Output directory & final filename
    output_dir = "/Users/vittorio/Desktop/FNO/output_directory"
    os.makedirs(output_dir, exist_ok=True)
    new_filename = "training_divergence_onehot.npz"
    new_full_path = os.path.join(output_dir, new_filename)

    # -------------------------------------------------
    # 2) Load geometry & build bc_cropped => shape (48,48,48)
    # -------------------------------------------------
    geom_fields = load_vtk_fields(geometry_file, use_cell_data=True)
    if "boundary_cells" not in geom_fields:
        raise KeyError(
            f"'boundary_cells' not found in {geometry_file}. "
            f"Available fields: {list(geom_fields.keys())}"
        )
    bc_flat = geom_fields["boundary_cells"]  # length should be 49^3
    bc_49 = reshape_field_3d(bc_flat, 49, 49, 49, ncomp=1, order="F")  # shape (49,49,49,1)
    bc_49 = bc_49[..., 0]  # squeeze to shape (49,49,49)
    # Crop the outer ghost layer to get a (48,48,48) array
    bc_cropped = bc_49[1:49, 1:49, 1:49]

    # ----------------------------------------------
    # 3) Build the velocity/pressure dataset, appending bc
    # ----------------------------------------------
    # Note: here the file patterns include subdirectories "intermediate" and "final".
    inputs_list, targets_list = build_training_dataset(
        directory=directory,
        tsteps=tsteps,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        bc_cropped=bc_cropped,
        intermediate_pattern="intermediate/roomVentilation_intermediate_{:d}.vtk",
        final_pattern="final/roomVentilation_{:d}.vtk",
    )

    if len(inputs_list) == 0:
        print("No data loaded (inputs_list is empty). Check your file paths or time steps.")
        return

    # Convert lists to arrays
    inputs_array = np.stack(inputs_list, axis=0)    # shape: [num_samples, Nx, Ny, Nz, 4]
    targets_array = np.stack(targets_list, axis=0)    # shape: [num_samples, Nx, Ny, Nz, 1]

    # ----------------------------------------------
    # 4) Process the dataset in memory:
    #    - Compute divergence from the velocity channels (channels 0,1,2)
    #    - Convert the boundary code (channel 3) to one-hot encoding
    # ----------------------------------------------
    # Extract velocity components (u, v, w)
    u_vals = inputs_array[..., 0]  # shape: [num_samples, Nx, Ny, Nz]
    v_vals = inputs_array[..., 1]
    w_vals = inputs_array[..., 2]

    # Compute spatial gradients along x, y, and z
    du_dx = np.gradient(u_vals, axis=1)  # derivative with respect to x
    dv_dy = np.gradient(v_vals, axis=2)  # derivative with respect to y
    dw_dz = np.gradient(w_vals, axis=3)  # derivative with respect to z

    # Compute divergence: du/dx + dv/dy + dw/dz
    div_vals = du_dx + dv_dy + dw_dz   # shape: [num_samples, Nx, Ny, Nz]
    div_vals_5d = np.expand_dims(div_vals, axis=-1)  # shape: [num_samples, Nx, Ny, Nz, 1]

    # Extract boundary code (channel 3) and convert to one-hot:
    # Assumption: boundary code 0 = free, 1 = wall, 2 = inlet.
    bc_vals = inputs_array[..., 3]  # shape: [num_samples, Nx, Ny, Nz]
    bc_free  = (bc_vals == 0).astype(np.float32)
    bc_wall  = (bc_vals == 1).astype(np.float32)
    bc_inlet = (bc_vals == 2).astype(np.float32)
    # Expand dimensions to 5D
    bc_free_5d  = np.expand_dims(bc_free, axis=-1)
    bc_wall_5d  = np.expand_dims(bc_wall, axis=-1)
    bc_inlet_5d = np.expand_dims(bc_inlet, axis=-1)

    # Concatenate divergence and one-hot boundaries into final input channels:
    # Final channels: 0 => divergence, 1 => free, 2 => wall, 3 => inlet.
    new_data = np.concatenate([
        div_vals_5d,     # divergence, shape: [..., 1]
        bc_free_5d,      # free boundary
        bc_wall_5d,      # wall boundary
        bc_inlet_5d      # inlet boundary
    ], axis=-1)
    # new_data shape: [num_samples, Nx, Ny, Nz, 4]

    # ----------------------------------------------
    # 5) Save the final dataset to disk
    # ----------------------------------------------
    # The saved file will contain:
    #   "inputs"  => new_data (divergence + one-hot boundary channels)
    #   "targets" => original pressure data (shape: [num_samples, Nx, Ny, Nz, 1])
    np.savez(new_full_path, inputs=new_data, targets=targets_array)

    print(f"Saved final dataset to {new_full_path}")
    print(f"Final input shape = {new_data.shape} (divergence, free, wall, inlet)")
    print(f"Final target shape = {targets_array.shape} (pressure)")

if __name__ == "__main__":
    main()
