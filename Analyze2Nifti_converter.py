import os
import nibabel as nb

'''
    This converter script contains functions for proper conversion of MR files from 
    Analyze 7.5 format to nifti.
    See the jupyter notebook(Analyze2Nifti_plus_orientationsStudy.ipynb) for the development steps and more info
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
    dirName = '/Users/bthomas/Downloads/Pathto/RootDirectory/'

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    numfiles = 0
    # Print the files
    for elem in listOfFiles:
        if elem.endswith(".img"):
            print('Reading', elem)
            img = nb.load(elem)
            numfiles += 1
            print('Converting.....')
            canonical_img = nb.as_closest_canonical(img)
            print('Now Saving', elem.replace('.img', '.nii.gz'), '\n')
            nb.save(canonical_img, elem.replace('.img', '.nii.gz'))
    print(numfiles, 'files converted')

    # Another way(os.walk) to get the list of all files in directory tree at given path


#     listOfFiles = list()
#     for (dirpath, dirnames, filenames) in os.walk(dirName):
#         listOfFiles += [os.path.join(dirpath, file) for file in filenames]

#     y=0
#     # Print the files
#     for elem in listOfFiles:
#         print(elem)
#         y+=1

#     print(y)

if __name__ == '__main__':
    main()