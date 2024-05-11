# Just KeepLargestConnectedComponentOnly
# Post process the Final outputs 

import os
import nibabel as nb
import SimpleITK as sitk
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import KeepLargestConnectedComponent
# import warnings
# warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


'''
    Postprocess and keep only the largest component
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles        
def main():
    exptfolder = "m15nm1m2m3m4"
    dirName = f'/Users/bthomas/Downloads/data_for_Belvin/midl24/{exptfolder}/seg_dice/before_cc'
    
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    numfiles=0
   
    # Print the files
    for elem in listOfFiles:
        if elem.endswith("T2W.nii.gz"):
            print('Reading', elem)
            filename = elem[80:87]
            im=sitk.GetArrayFromImage(sitk.ReadImage(elem))
            transform = KeepLargestConnectedComponent(applied_labels = [5],is_onehot=False,independent=True)  # Bgd and psoas are purposefully ignored.
            im_array = transform(im)
            im_keepcc = sitk.GetImageFromArray(im_array)
            sitk.WriteImage(im_keepcc,f'{dirName}/../after_cc/seg{filename}_T2W.nii.gz')
            numfiles+=1
#             print('Saving.....')
    print(numfiles, 'files converted to ITK-readable with only LargestConnectedComponent')    
        
if __name__ == '__main__':
    main()
