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
    """
    if field_array.ndim == 1:
        field_array = field_array[:, None]  # [Npoints, 1]

    Npoints = Nx * Ny * Nz
    if field_array.shape[0] != Npoints:
        raise ValueError(
            f"Field has {field_array.shape[0]} points, expected {Npoints} "
            f"for Nx={Nx},Ny={Ny},Nz={Nz}."
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
      - 'intermediate' file => velocity_intermediate
      - 'final' file => pressure
    Reshapes them into 4D arrays:
      velocity: [Nx, Ny, Nz, 3]
      pressure: [Nx, Ny, Nz, 1]
    Then appends the boundary code as the 4th channel => [Nx, Ny, Nz, 4] for velocity+bc.

    bc_cropped: shape (Nx, Ny, Nz) with boundary codes {0,1,2,...}

    Returns:
      inputs_list: list of arrays [Nx, Ny, Nz, 4]
      targets_list: list of arrays [Nx, Ny, Nz, 1]
    """
    inputs_list = []
    targets_list = []

    bc_expanded = bc_cropped[..., None]  # => (Nx,Ny,Nz,1)

    for t in tsteps:
        inter_file = os.path.join(directory, intermediate_pattern.format(t))
        if not os.path.isfile(inter_file):
            print(f"Warning: file not found {inter_file}, skipping timestep {t}.")
            continue
        inter_fields = load_vtk_fields(inter_file, use_cell_data=False)

        final_file = os.path.join(directory, final_pattern.format(t))
        if not os.path.isfile(final_file):
            print(f"Warning: file not found {final_file}, skipping timestep {t}.")
            continue
        final_fields = load_vtk_fields(final_file, use_cell_data=False)

        if "velocity_intermediate" not in inter_fields:
            raise KeyError(
                f"'velocity_intermediate' not found in {inter_file}. "
                f"Fields available: {list(inter_fields.keys())}"
            )
        vel_intermediate = inter_fields["velocity_intermediate"]

        if "p_first" not in final_fields:
            raise KeyError(
                f"'pressure' not found in {final_file}. "
                f"Fields available: {list(final_fields.keys())}"
            )
        pressure = final_fields["p_first"]

        vel_reshaped = reshape_field_3d(vel_intermediate, Nx, Ny, Nz, ncomp=3, order="F")
        press_reshaped = reshape_field_3d(pressure, Nx, Ny, Nz, ncomp=1, order="F")

        vel_plus_bc = np.concatenate([vel_reshaped, bc_expanded], axis=-1)

        inputs_list.append(vel_plus_bc)     # shape => [Nx,Ny,Nz,4]
        targets_list.append(press_reshaped) # shape => [Nx,Ny,Nz,1]

        print(
            f"Timestep {t}: velocity shape={vel_reshaped.shape}, bc shape={bc_expanded.shape}, "
            f"=> input shape={vel_plus_bc.shape}, pressure shape={press_reshaped.shape}"
        )

    return inputs_list, targets_list


def process_simulation_folder(
    folder_path,
    tsteps,
    Nx,
    Ny,
    Nz,
    output_dir,
    output_filename="training_poisson_four_channels.npz"
):
    """
    Process a single simulation folder, which should contain
    - a 'final' subfolder with 'roomVentilation_geometry.vtk'
    - an 'intermediate' subfolder with velocity files
    - a 'final' subfolder with pressure files
    """

    # 1) Verify subfolders exist
    final_folder = os.path.join(folder_path, "final")
    intermediate_folder = os.path.join(folder_path, "intermediate")
    if not os.path.isdir(final_folder) or not os.path.isdir(intermediate_folder):
        print(f"Skipping '{folder_path}' -- missing 'final' or 'intermediate' subfolder.")
        return

    geometry_file = os.path.join(final_folder, "roomVentilation_geometry.vtk")
    if not os.path.isfile(geometry_file):
        print(f"Skipping '{folder_path}' -- no geometry file found.")
        return

    # 2) Load geometry & build bc_cropped => (Nx,Ny,Nz)
    geom_fields = load_vtk_fields(geometry_file, use_cell_data=True)
    if "boundary_cells" not in geom_fields:
        print(f"Skipping '{folder_path}' -- 'boundary_cells' not found in geometry file.")
        return

    bc_flat = geom_fields["boundary_cells"]  # length => likely (Nx+1)*(Ny+1)*(Nz+1)
    # Adjust the shape if your actual geometry is e.g. 49^3
    bc_49 = reshape_field_3d(bc_flat, Nx+1, Ny+1, Nz+1, ncomp=1, order="F")
    bc_49 = bc_49[..., 0]  # => (Nx+1,Ny+1,Nz+1)

    # Crop ghost layer => shape (Nx,Ny,Nz)
    bc_cropped = bc_49[1: Nx+1, 1: Ny+1, 1: Nz+1]

    # 3) Build velocity/pressure dataset
    inputs_list, targets_list = build_training_dataset(
        directory=folder_path,   # <== folder_path that has "intermediate", "final"
        tsteps=tsteps,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        bc_cropped=bc_cropped,
        intermediate_pattern="intermediate/roomVentilation_intermediate_{:d}.vtk",
        final_pattern="final/roomVentilation_{:d}.vtk",
    )

    if len(inputs_list) == 0:
        print(f"No data loaded in '{folder_path}' (inputs_list is empty).")
        return

    # 4) Convert to final arrays
    inputs_array = np.stack(inputs_list, axis=0)   # => [num_samples, Nx, Ny, Nz, 4]
    targets_array = np.stack(targets_list, axis=0) # => [num_samples, Nx, Ny, Nz, 1]

    # 5) Save into a subfolder in output_dir
    folder_name = os.path.basename(folder_path.strip("/"))
    subfolder_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(subfolder_output_dir, exist_ok=True)

    output_path = os.path.join(subfolder_output_dir, output_filename)
    np.savez(output_path, inputs=inputs_array, targets=targets_array)

    print(f"Saved dataset for folder '{folder_path}' to {output_path}")
    print(f"Final input shape = {inputs_array.shape} (u,v,w,bc)")
    print(f"Final target shape = {targets_array.shape} (pressure)")


def main():
    # ------------------------------------------------
    # 1) Global parameters
    # ------------------------------------------------
    base_directory = "/Users/vittorio/Desktop/GSoC_FFD_PC/simulations_output/"
    # For each folder, we look for 'intermediate/roomVentilation_intermediate_{t}.vtk'
    # and 'final/roomVentilation_{t}.vtk'. Suppose all subfolders share the same timesteps:
    tsteps = range(1, 800)

    # Domain size for interior data
    Nx, Ny, Nz = 48, 48, 48

    # Directory where final datasets are stored
    output_dir = "/Users/vittorio/Desktop/GSoC_FFD_PC/all_output_npz"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------
    # 2) Loop over subfolders in base_directory
    # ------------------------------------------------
    for subfolder in sorted(os.listdir(base_directory)):
        folder_path = os.path.join(base_directory, subfolder)
        # Check if it's really a directory
        if not os.path.isdir(folder_path):
            continue

        # Check if we have already processed this subfolder
        # by seeing if the corresponding subfolder exists in the output_dir
        subfolder_output_dir = os.path.join(output_dir, subfolder)
        if os.path.isdir(subfolder_output_dir):
            print(f"Skipping '{folder_path}' -- it has already been processed.")
            continue

        print(f"\n--- Processing folder: {folder_path} ---")
        process_simulation_folder(
            folder_path=folder_path,
            tsteps=tsteps,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
