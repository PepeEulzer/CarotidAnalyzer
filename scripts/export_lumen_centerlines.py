import os
import shutil
import json
import vtk
from glob import glob

"""
Script to export lumen .stl and centerline .obj files for all patients that have both
for use in CFD modelling.
Also exports a .csv with meta information on each stenosis.
Requires VTK >= 8.2 (obj writer).
"""

src_folder = 'C:/Users/Pepe Eulzer/Nextcloud/carotis_data'
dst_folder = 'C:/Users/Pepe Eulzer/Desktop/daten_kevin'

metadata = [["ID", "Stenosis Degree", "Min Diameter Position", "Min Diameter Normal", "Ref Diameter Position", "Ref Diameter Normal"]]

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

            # add to metadata
            meta_filepath = os.path.join(patient_src_folder, "models", (patient_id + "_" + side + "_meta.txt"))
            meta_entry = [patient_id + "_" + side, [], [], [], [], []]
            if os.path.exists(meta_filepath):
                try:
                    with open(meta_filepath, 'r') as f:
                        d = json.load(f)
                        meta_entry = [patient_id + "_" + side,
                              d["stenosis_degree"],
                              d["stenosis_min_p"],
                              d["stenosis_min_n"],
                              d["poststenotic_p"],
                              d["poststenotic_n"]]
                except:
                    print("WARNING! Classifier module could not load " + meta_filepath)
            metadata.append(meta_entry)

# save metadata csv
csv_path = os.path.join(dst_folder, "stenosis_metadata.csv")
with open(csv_path, 'w') as f:
    for line in metadata:
        for item in line:
            f.write(str(item))
            f.write(';')
        f.write('\n')

print("Done!")