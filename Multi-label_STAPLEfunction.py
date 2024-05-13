# A Multi-label STAPLE function to use sitk.MultiLabelSTAPLEImageFilter

# help(sitk.MultiLabelSTAPLEImageFilter)

import SimpleITK as sitk

def staple_multi_label(segmentation_files, label_undecided_pixel):
#     sitk_segmentations = [sitk.GetImageFromArray(x) for x in segmentations]
    sitk_segmentations = [sitk.GetImageFromArray(x) for x in segmentation_files]
    MLSTAPLE = sitk.MultiLabelSTAPLEImageFilter()
    MLSTAPLE.SetLabelForUndecidedPixels(label_undecided_pixel)
    msk = MLSTAPLE.Execute(sitk_segmentations)
    sitk.WriteImage(msk,'/Users/bthomas/Downloads/MR_Sarcopaenia/STAPLEoutputs/StaplemaskAllCNNsplusTFRs.nii.gz')
    return sitk.GetArrayFromImage(msk)

# Example usage:  sitk.MultiLabelSTAPLEImageFilter


segmentation_file_names = ["/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_Adam.nii.gz", 
                          "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_Anthony.nii.gz",
                          "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_Fatima.nii.gz",
                          "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_Pardis.nii.gz",
                          "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_SophiaMoeko.nii.gz",
                          "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_Gabie.nii",
                          "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_Hadil.nii",
                          "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_James_200224.nii",
                           "/Users/bthomas/Downloads/MR_Sarcopaenia/ExpertAnnotations/seg_Manjiri.nii"]

segmentation_files = [sitk.GetArrayFromImage(sitk.ReadImage(file_name, sitk.sitkUInt8)) for file_name in segmentation_file_names]
mask = staple_multi_label(segmentation_files,6)