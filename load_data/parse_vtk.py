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

    # Expand bc_cropped to shape (Nx, Ny, Nz, 1) so we can concatenate easily
    bc_expanded = bc_cropped[..., None]  # => (48,48,48,1)

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

        # 3) Extract velocity_intermediate => shape [Npoints,3]
        if "velocity_intermediate" not in inter_fields:
            raise KeyError(
                f"'velocity_intermediate' not found in {inter_file}. "
                f"Fields available: {list(inter_fields.keys())}"
            )
        vel_intermediate = inter_fields["velocity_intermediate"]

        # 4) Extract pressure => shape [Npoints,]
        if "pressure" not in final_fields:
            raise KeyError(
                f"'pressure' not found in {final_file}. "
                f"Fields available: {list(final_fields.keys())}"
            )
        pressure = final_fields["pressure"]

        # 5) Reshape
        vel_reshaped = reshape_field_3d(vel_intermediate, Nx, Ny, Nz, ncomp=3, order="F")
        press_reshaped = reshape_field_3d(pressure, Nx, Ny, Nz, ncomp=1, order="F")

        # 6) Concat bc => shape [Nx, Ny, Nz, 4]
        # axis=-1 means we add the boundary code as an additional channel at the end
        vel_plus_bc = np.concatenate([vel_reshaped, bc_expanded], axis=-1)

        inputs_list.append(vel_plus_bc)     # shape => [Nx,Ny,Nz,4]
        targets_list.append(press_reshaped) # shape => [Nx,Ny,Nz,1]

        print(
            f"Timestep {t}: velocity shape={vel_reshaped.shape}, bc shape={bc_expanded.shape}, "
            f" => input shape={vel_plus_bc.shape}, pressure shape={press_reshaped.shape}"
        )

    return inputs_list, targets_list


def main():
    # -----------------------------
    # 1) Setup parameters & paths
    # -----------------------------
    directory = "/Users/vittorio/Desktop/GSoC_FFD_PC/simulations_output/0/"
    tsteps = range(1, 100)

    # Our interior domain for velocity & pressure is 48^3
    Nx, Ny, Nz = 48, 48, 48

    # We'll read geometry => 49^3 cell data (DIMENSIONS 50 50 50)
    geometry_file = os.path.join(directory, "final", "roomVentilation_geometry.vtk")

    # Output
    output_dir = "/Users/vittorio/Desktop/GSoC_FFD_PC/output_directory"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "training_poisson_four_channels.npz"
    output_path = os.path.join(output_dir, output_filename)

    # -------------------------------------------------
    # 2) Load geometry & build bc_cropped => (48,48,48)
    # -------------------------------------------------
    geom_fields = load_vtk_fields(geometry_file, use_cell_data=True)
    if "boundary_cells" not in geom_fields:
        raise KeyError(
            f"'boundary_cells' not found in {geometry_file}. "
            f"Fields in geometry: {list(geom_fields.keys())}"
        )

    bc_flat = geom_fields["boundary_cells"]  # length => 49^3 = 117,649
    bc_49 = reshape_field_3d(bc_flat, 49, 49, 49, ncomp=1, order="F")  # => shape (49,49,49,1)
    bc_49 = bc_49[..., 0]  # => (49,49,49)

    # Crop outer ghost layer => shape (48,48,48)
    bc_cropped = bc_49[1:49, 1:49, 1:49]

    # ----------------------------------------------
    # 3) Build velocity/pressure dataset, appending bc
    # ----------------------------------------------
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

    # Convert to final arrays
    inputs_array = np.stack(inputs_list, axis=0)   # => [num_samples, Nx, Ny, Nz, 4]
    targets_array = np.stack(targets_list, axis=0) # => [num_samples, Nx, Ny, Nz, 1]

    # ----------------------------------------------
    # 4) Save everything
    # ----------------------------------------------
    np.savez(output_path, inputs=inputs_array, targets=targets_array)

    print(f"Saved dataset to {output_path}")
    print(f"Final input shape = {inputs_array.shape} (u,v,w,bc)")
    print(f"Final target shape = {targets_array.shape} (pressure)")


if __name__ == "__main__":
    main()
