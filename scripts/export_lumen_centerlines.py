import os
import shutil
import vtk
from glob import glob

"""
Script to export lumen .stl and centerline .obj files for all patients that have both
for use in CFD modelling.
Requires VTK >= 8.2 (obj writer).
"""

src_folder = 'C:/Users/Pepe Eulzer/Nextcloud/daten_wei_chan'
dst_folder = 'C:/Users/Pepe Eulzer/Desktop/daten_kevin'

# iterate patients
for patient_src_folder in glob(os.path.join(src_folder, "patient*")):
    patient_id = os.path.basename(patient_src_folder)
    # iterate left/right
    for side in ['left', 'right']:
        lumen_src_filepath = os.path.join(patient_src_folder, "models", (patient_id + "_" + side + "_lumen.stl"))
        centerlines_src_filepath = os.path.join(patient_src_folder, "models", patient_id + "_" + side + "_lumen_centerlines.vtp")
        
        # copy files if both lumen and centerline were found
        if os.path.exists(lumen_src_filepath) and os.path.exists(centerlines_src_filepath):
            print("Copying " + patient_id + "...")
            patient_dst_folder = os.path.join(dst_folder, patient_id)
            if not os.path.exists(patient_dst_folder):
                os.makedirs(patient_dst_folder)
            
            # copy lumen
            lumen_dst_filepath = os.path.join(patient_dst_folder, patient_id + "_" + side + "_lumen.stl")
            shutil.copyfile(lumen_src_filepath, lumen_dst_filepath)

            # copy centerline, convert .vtp -> .obj
            centerlines_dst_filepath = os.path.join(patient_dst_folder, patient_id + "_" + side + "_lumen_centerlines.obj")
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(centerlines_src_filepath)
            reader.Update()
            writer = vtk.vtkOBJWriter()
            writer.SetInputData(reader.GetOutput())
            writer.SetFileName(centerlines_dst_filepath)
            writer.Write()

print("Done!")