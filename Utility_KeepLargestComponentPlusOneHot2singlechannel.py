# With Keep LargestConnectedComponentOnly
# Post process the MONAI outputs (Convert Monai labels (Multi-channel One-hot format)to labels readable in ITK-SNAP (Single Channel labelled 0 to 5))

import os
import nibabel as nb
import SimpleITK as sitk
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import KeepLargestConnectedComponent

'''
    For the given path, get the List of all files in the directory tree 
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
    
    dirName = '/Users/bthomas/Downloads/temp_for_Belvin/Files_extracted/seg/outs_fromMonai'
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    numfiles=0
    groups=6
    # Print the files
    for elem in listOfFiles:
        if elem.endswith("T2W_seg.nii.gz"):
            print('Reading', elem)
            filename = elem[76:83]
            im=sitk.GetArrayFromImage(sitk.ReadImage(elem))
            transform = KeepLargestConnectedComponent(applied_labels = [1,2,3,5],is_onehot=True,independent=False)  # Bgd and psoas are purposefully ignored.
            im_array = transform(im)
            labelled = np.zeros([im_array.shape[1],im_array.shape[2],im_array.shape[3]])
            for i in range(groups):
                ilabel = np.squeeze(im_array[i,:,:,:])
                labelled[ilabel == 1] = i
            im_labelled = sitk.GetImageFromArray(labelled)
            sitk.WriteImage(im_labelled,f'{dirName}/../outs_ForITK/seg{filename}_T2W.nii.gz')
            numfiles+=1
            print('Saving.....')
    print(numfiles, 'files converted to ITK-readable with only LargestConnectedComponent')    
        
if __name__ == '__main__':
    main()
