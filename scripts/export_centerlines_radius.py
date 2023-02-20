import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from glob import glob
"""
Script to export centerline .obj and radius .txt files for use in clustering.
Requires VTK >= 8.2 (obj writer).
"""

src_folder = 'E:/carotis_data_backup02'
dst_folder = 'C:/Users/Pepe Eulzer/Nextcloud/carotis_data/centerlines_radii_landau'

# iterate patients
for patient_src_folder in glob(os.path.join(src_folder, "patient*")):
    patient_id = os.path.basename(patient_src_folder)
    # iterate left/right
    for side in ['left', 'right']:
        centerlines_src_filepath = os.path.join(patient_src_folder, "models", patient_id + "_" + side + "_lumen_centerlines.vtp")

        # export if file exists
        if os.path.exists(centerlines_src_filepath):
            print("Copying " + patient_id + "...")
            patient_dst_folder = os.path.join(dst_folder, patient_id)
            if not os.path.exists(patient_dst_folder):
                os.makedirs(patient_dst_folder)
            
            # copy centerline, convert .vtp -> .obj
            centerlines_dst_filepath = os.path.join(patient_dst_folder, patient_id + "_" + side + "_lumen_centerlines.obj")
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(centerlines_src_filepath)
            reader.Update()
            writer = vtk.vtkOBJWriter()
            writer.SetInputData(reader.GetOutput())
            writer.SetFileName(centerlines_dst_filepath)
            writer.Write()

            # copy radius array -> .txt
            radius_dst_filepath = os.path.join(patient_dst_folder, patient_id + "_" + side + "_radii.txt")
            radii_flat = reader.GetOutput().GetPointData().GetArray('MaximumInscribedSphereRadius')
            r = vtk_to_numpy(radii_flat)
            np.savetxt(radius_dst_filepath, r)

print("Done!")