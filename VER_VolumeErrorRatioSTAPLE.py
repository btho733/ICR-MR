# Volume Error Ratio using STAPLE stack (STAPLEMASK as ref volume)Full : All 6 tissue classes

import os
import nibabel as nb
import SimpleITK as sitk
import matplotlib.pyplot as plt
# %matplotlib notebook
import numpy as np
import pandas as pd
from monai.transforms import AsDiscrete
import torch
from monai.metrics import DiceMetric
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

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
    
    exptfolder = "m16unetrmean"
    root_dir = "/Users/bthomas/Downloads/data_for_Belvin/midl24/staple/Annotations"
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(root_dir)
    numfiles=0
    df1 = pd.DataFrame(columns=['Comparison Vs STAPLE','Bgd','Cavityexcluded','SubcutFat','IntFat','Psoas','Muscle','totalfat','totalmuscle'])
    # Print the files
    numAnnotator = 0
    allraters = []
    allautos = []
    for elem in listOfFiles:
        if elem.endswith(".nii.gz"):
            imname = elem[67:-2]
            im = sitk.ReadImage(elem)
            print('Reading', imname)
            ylabel = sitk.GetArrayFromImage(sitk.ReadImage('/Users/bthomas/Downloads/data_for_Belvin/midl24/staple/Staplemask.nii.gz', sitk.sitkInt16))
            ypred = sitk.GetArrayFromImage(sitk.ReadImage(f'/Users/bthomas/Downloads/data_for_Belvin/midl24/staple/Annotations/{imname}gz', sitk.sitkInt16))
            
            refbgd = ylabel==0
            refcavity = ylabel==1
            refsubcutfat = ylabel == 2
            refintfat = ylabel == 3
            refpsoas = ylabel == 4
            refmuscle = ylabel == 5
            reftotalfat = refsubcutfat + refintfat
            reftotalmuscle = refpsoas + refmuscle
            
            segbgd = ypred==0
            segcavity = ypred==1
            segsubcutfat = ypred == 2
            segintfat = ypred == 3
            segpsoas = ypred == 4
            segmuscle = ypred == 5
            segtotalfat = segsubcutfat + segintfat
            segtotalmuscle = segpsoas + segmuscle
            
            ver_bgd = (np.sum(refbgd) -np.sum(segbgd))/np.sum(refbgd)
            ver_cavity = (np.sum(refcavity) - np.sum(segcavity))/np.sum(refcavity)
            ver_subcutfat = (np.sum(refsubcutfat) - np.sum(segsubcutfat))/np.sum(refsubcutfat)
            ver_intfat = (np.sum(refintfat) - np.sum(segintfat))/np.sum(refintfat)
            ver_psoas = (np.sum(refpsoas)  - np.sum(segpsoas))/np.sum(refpsoas) 
            ver_muscle = (np.sum(refmuscle) - np.sum(segmuscle))/np.sum(refmuscle)
            ver_totalfat = (np.sum(reftotalfat)  - np.sum(segtotalfat))/np.sum(reftotalfat) 
            ver_totalmuscle = (np.sum(reftotalmuscle) - np.sum(segtotalmuscle))/np.sum(reftotalmuscle)

#             df1append = pd.DataFrame(data={'Comparison':'Auto Vs AvgRater','Bgd': ver_bgd,'Cavityexcluded':ver_cavity,\
#                                       'SubcutFat':ver_subcutfat,'IntFat':ver_intfat,'Psoas':ver_psoas,\
#                                         'Muscle':ver_muscle,'Mean':np.mean([ver_bgd,ver_cavity,ver_subcutfat,ver_intfat,ver_psoas,ver_muscle]),\
#                                      'totalfat':ver_totalfat,'totalmuscle':ver_totalmuscle},index=['Filename'])
            
            if imname.startswith("seg"):
                numAnnotator+=1
                newrater = {'Comparison Vs STAPLE':f'Rater_{numAnnotator}','Bgd': ver_bgd,'Cavityexcluded':ver_cavity,\
                                      'SubcutFat':ver_subcutfat,'IntFat':ver_intfat,'Psoas':ver_psoas,\
                                        'Muscle':ver_muscle,'totalfat':ver_totalfat,'totalmuscle':ver_totalmuscle}
                
                allraters.append(newrater)
            else: 
                newauto = {'Comparison Vs STAPLE':f'{imname[8:11]}','Bgd': ver_bgd,'Cavityexcluded':ver_cavity,\
                                      'SubcutFat':ver_subcutfat,'IntFat':ver_intfat,'Psoas':ver_psoas,\
                                        'Muscle':ver_muscle,'totalfat':ver_totalfat,'totalmuscle':ver_totalmuscle}
                allautos.append(newauto)
                 
            numfiles+=1          
    print(numfiles, 'files read')
    df1 = df1.append(allraters,ignore_index=True) 
    df1 = df1.append(allautos,ignore_index=True) 
    
#     mean_row = df1.iloc[:, 1:8].mean().round(4)
#     std_row = df1.iloc[:, 1:8].std().round(4)
   

#     formatted_values = []
#     for i in range(len(mean_row)):
#         formatted_value = f"{mean_row[i]:.3f} ± {std_row[i]:.2f}"
#         formatted_values.append(formatted_value)

#     df2 = pd.DataFrame(data={'Filename':'Mean ± SD','Bgd': formatted_values[0],'Cavityexcluded':formatted_values[1],\
#                                       'SubcutFat':formatted_values[2],'IntFat':formatted_values[3],'Psoas':formatted_values[4],\
#                                         'Muscle':formatted_values[5],'Mean':formatted_values[6]},index=[''])
#     df3= pd.DataFrame(data={'Filename':'Mean','Bgd': mean_row[0],'Cavityexcluded':mean_row[1],\
#                                       'SubcutFat':mean_row[2],'IntFat':mean_row[3],'Psoas':mean_row[4],\
#                                         'Muscle':mean_row[5],'Mean':mean_row[6]},index=[''])
    
#     finaldf = pd.concat([df1],axis=0,ignore_index=True)
#     df1_sorted = df1.sort_values(by='Muscle')
#     df1_sorted = df1.sort_values(by='IntFat', ascending=False)
    display(df1)
    
#     with pd.ExcelWriter(f'{root_dir}/../VERstaple_{exptfolder}.xlsx') as writer:
#         df.to_excel(writer, sheet_name=f'{exptfolder}_full')
if __name__ == '__main__':
    main()