import os
import nibabel as nb
import numpy as np
import pandas as pd
import SimpleITK as sitk


"""Tools for repeatability calculations"""

import os
from collections import OrderedDict


import numpy as np
%matplotlib notebook
import matplotlib.pyplot as pl
import scipy.stats
import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

'''Compute all the repeatability stats (ICC,RC(symmetric), LoA(Asymmetric),wCV etc) and Plot B-A plots'''



def plot_BA_in_axis(x1, x2, z1, z2, fname, ax, parameter_name='data', units=r'AU'):
    x1 = np.array(x1)
    x2 = np.array(x2)
#     x1 = np.log((x1)
#     x2 = np.log(x2)
    x = (x1 + x2) / 2
    d = x2 - x1
    N = x1.size

#     stats = repeatability_stats(x1, x2,False)
#     RC = stats['original']['r']
#     RC_CI = stats['original']['r_CI']
#     icc = stats["original"]['ICC']
#     icc_CI = stats["original"]['ICC_CI']
    
    stats = repeatability_stats(x1, x2,True)
    RC = stats['log']['r']
    RC_CI = stats['log']['r_CI']
    icc = stats["log"]['ICC']
    icc_CI = stats["log"]['ICC_CI']
    
    z1 = np.array(z1)
    z2 = np.array(z2)
   
    x44 = []
    d44 = []
    x48 = []
    d48 = []
    x84 = []
    d84 = []
    x88 = []
    d88 = []
    
    
    for i in range(0, len(x)):
        if z1[i]==4 and z2[i]==4:
            x44.append(x[i])
            d44.append(d[i])
        elif z1[i]==4 and z2[i]==8:
            x48.append(x[i])
            d48.append(d[i])
        elif z1[i]==8 and z2[i]==4:
            x84.append(x[i])
            d84.append(d[i])
        elif z1[i]==8 and z2[i]==8:
            x88.append(x[i])
            d88.append(d[i])
        else:
            print('There is something other than z = 4 or z = 8')
   
    ax.scatter(x44, d44, color = 'red', marker= 'v', label='z = 4mm')
#     ax.scatter(x48, d48, color = 'purple', marker= '*', label='z1=4  z2=8')
#     ax.scatter(x84, d84, color = 'blue', marker= '*', label='z1=8  z2=4')
    ax.scatter(x88, d88, color = 'green', marker= 'o', label='z = 8mm')
    ax.set_title(r"%s" % parameter_name, fontsize = 12.0)
    ax.set_xlabel("$(x_{2} + x_{1}) / 2$ litres", fontsize = 10.0)
    ax.set_ylabel("$x_{2} - x_{1}$ ", fontsize = 10.0)

    Dx = 0.1 * (x.max() - x.min())
    x_ = np.linspace(x.min() - Dx, x.max() + Dx, 100)

    ax.plot(x_, np.repeat(RC, x_.size), 'k--', lw=3.0)
    ax.plot(x_, np.repeat(-1.0 * RC, x_.size), 'k--', lw=3.0)
    ax.set_xlim((x_.min(), x_.max()))
    ylim = 1.5 * np.abs(np.r_[RC, d]).max()
    ax.set_ylim((-ylim, ylim))
#     ax.text(0.97, 0.97,
#             "RC = %.2f (%.2f, %.2f)\nICC = %.2f (%.2f, %.2f)" % (RC, RC_CI[0], RC_CI[1], icc, icc_CI[0], icc_CI[1]),
#             horizontalalignment='right', verticalalignment='top', fontsize=13.0, transform=ax.transAxes)
    ax.legend(loc='lower left')
    for i in range(len(x)):
        ax.text(x[i], d[i], str(fname[i]), color="black", fontsize=5)
    
    


def repeatability_stats(x1, x2, calc_log):
    # Takes in two vectors of data and provides repeatability statistics

    # Make sure they are numpy vectors and have the same shape
    x1 = np.array(x1)
    x2 = np.array(x2)
    if np.size(x1.shape) > 1 or np.size(x2.shape) > 1:
        raise Exception("x1 and x2 must be vectors")
    if not np.array_equal(x1.shape, x2.shape):
        raise Exception("x1 and x2 must have same shape!")

    # The size of the data
    N = float(x1.size)

    # First calcaulte the repeatability using the un-logged data
    # The difference between data points
    d = x2 - x1

    # The overall mean and within subject means (M and m respectively)
    m = (x1 + x2) / 2
    M = np.mean(m)

    ddof = 0 # For now the degrees of freedom is 0 (assume x1 and x2 are not biased)

    # Inverse chi-square values for CI calcaultion
    ichi_025 = scipy.stats.chi2.ppf(0.025, N)
    ichi_975 = scipy.stats.chi2.ppf(0.975, N)

    sw = np.sqrt(np.sum(d**2)/(2.0 * (N-ddof)))
    sw_CI = np.array([sw * np.sqrt(N / ichi_975), sw * np.sqrt(N / ichi_025)])
    sw_CoV = sw/M * 100.0
    # Should we have a sw_COV_CI here?  Need to determine what this would be given error in M!

    r = 1.96 * np.sqrt(2) * sw
    r_CI = 1.96 * np.sqrt(2) * sw_CI
    r_CoV = r/M * 100.0

    # Between-subject mean squares and within-subject mean squares respectively
    BMS = (2.0 / (N-ddof)) * np.sum((m - M) ** 2)
    WMS = np.sum((x1 - m) ** 2 + (x2 - m) ** 2) / (N-ddof)

    sb = np.sqrt((BMS-WMS)/2)
    sb_CoV = sb/M * 100.0

    ICC = sb**2 / (sb**2 + sw**2)
    F0 = BMS / WMS
    FU = F0 * scipy.stats.f.ppf(0.975, N, N - 1)
    FL = F0 / scipy.stats.f.ppf(0.975, N - 1, N)
    ICC_CI = [(FL - 1) / (FL + 1), (FU - 1) / (FU + 1)]

    stats = {}
    stats["original"] = {'sw':sw, 'sw_CI':sw_CI, 'sw_CoV':sw_CoV,
                         'r':r, 'r_CI':r_CI, 'r_CoV':r_CoV,
                         'sb':sb, 'sb_CoV':sb_CoV,
                         'BMS':BMS, 'WMS':WMS, 'ICC':ICC, 'ICC_CI':ICC_CI}

    if calc_log:
        # Now perform for the logarithm of the data
        x1 = np.log(x1)
        x2 = np.log(x2)

        d = x2 - x1

        # The overall mean and within subject means (M and m respectively)
        m = (x1 + x2) / 2
        M = np.mean(m)

        sw = np.sqrt(np.sum(d ** 2) / (2.0 * (N - ddof)))
        sw_CI = np.array([sw * np.sqrt(N / ichi_975), sw * np.sqrt(N / ichi_025)])
        sw_CoV = np.sqrt(np.exp(sw ** 2) - 1) * 100.0
        sw_CoV_CI = np.sqrt(np.exp(sw_CI ** 2) - 1) * 100.0

        r = 1.96 * np.sqrt(2) * sw
        r_CI = 1.96 * np.sqrt(2) * sw_CI

        LoA = np.r_[np.exp(-1.96 * sw * np.sqrt(2)) - 1, np.exp(+1.96 * sw * np.sqrt(2)) - 1] * 100.
        LoA_CI = np.r_[np.exp(-1.96 * sw_CI[::-1] * np.sqrt(2)) - 1, np.exp(+1.96 * sw_CI * np.sqrt(2)) - 1] * 100.
        # Reverse the sign to ensure LoA CI in increasgin order.

        # Between-subject mean squares and within-subject mean squares respectively
        BMS = (2.0 / (N - ddof)) * np.sum((m - M) ** 2)
        WMS = np.sum((x1 - m) ** 2 + (x2 - m) ** 2) / (N - ddof)

        sb = np.sqrt((BMS - WMS) / 2)
        sb_CoV = np.sqrt(np.exp(sb ** 2) - 1) * 100.0

        ICC = sb ** 2 / (sb ** 2 + sw ** 2)
        F0 = BMS / WMS
        FU = F0 * scipy.stats.f.ppf(0.975, N, N - 1)
        FL = F0 / scipy.stats.f.ppf(0.975, N - 1, N)
        ICC_CI = [(FL - 1) / (FL + 1), (FU - 1) / (FU + 1)]

        stats['log'] = {'sw': sw, 'sw_CI': sw_CI, 'sw_CoV': sw_CoV, 'sw_CoV_CI':sw_CoV_CI,
                        'r': r, 'r_CI': r_CI,
                        'LoA':LoA, 'LoA_CI':LoA_CI,
                        'sb': sb, 'sb_CoV': sb_CoV,
                        'BMS': BMS, 'WMS': WMS, 'ICC': ICC, 'ICC_CI': ICC_CI}

    return stats

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
    exptfolder = 'm19swinvotex'
    psoasdata = 'okplusno'
    numcases = '49'
    dirName = f'/Users/bthomas/Downloads/data_for_Belvin/midl24/{exptfolder}/seg_v1v2/{psoasdata}/after_cc'
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    numfiles=0
    dfscan1 = pd.DataFrame(columns=['Patient','Visit','subcutfatvol','intfatvol','psoasvol','musclevol','totalfatvol','totalmusclevol', 'zdim'])
    dfscan2 = pd.DataFrame(columns=['Patient','Visit','subcutfatvol','intfatvol','psoasvol','musclevol','totalfatvol','totalmusclevol','zdim'])
    for elem in listOfFiles:
        if elem.endswith("T2W.nii.gz"):
#             print('Current file is ', elem)
            filename = elem[91:109]
            visitnum = int(elem[97])
            mask = sitk.ReadImage(elem)
            maskarray = sitk.GetArrayFromImage(mask)
            subcutfat = maskarray == 2
            intfat = maskarray == 3
            psoas = maskarray == 4
            muscle = maskarray == 5
            totalfat = subcutfat + intfat
            totalmuscle = psoas + muscle
            
            img = nb.load(f'/Users/bthomas/Downloads/data_for_Belvin/repeatability/forPlot/img/{filename}')
            zdim = np.round(img.header['pixdim'][3],3)
            voxelvol = np.prod(img.header['pixdim'][1:4])/(1e+6)
            
            subcutfatvol = np.sum(subcutfat) * voxelvol 
            intfatvol = np.sum(intfat) * voxelvol
            psoasvol = np.sum(psoas) * voxelvol
            musclevol = np.sum(muscle) * voxelvol
            totalfatvol = np.sum(totalfat) * voxelvol
            totalmusclevol = np.sum(totalmuscle) * voxelvol
            numfiles+=1
            if visitnum == 1:
                dfscan1 = dfscan1.append({'Patient':elem[91:95], 'Visit': visitnum,\
                            'subcutfatvol':subcutfatvol,'intfatvol':intfatvol,\
                            'psoasvol':psoasvol,'musclevol':musclevol,\
                            'totalfatvol':totalfatvol, 'totalmusclevol':totalmusclevol, 'zdim':zdim},ignore_index=True)
            elif visitnum == 2:
                dfscan2 = dfscan2.append({'Patient':elem[91:95], 'Visit': visitnum,\
                            'subcutfatvol':subcutfatvol,'intfatvol':intfatvol,\
                            'psoasvol':psoasvol,'musclevol':musclevol,\
                            'totalfatvol':totalfatvol, 'totalmusclevol':totalmusclevol, 'zdim':zdim},ignore_index=True)
            else:
                print('Scan is not from visit 1 or visit 2 ')
                                          
    print(numfiles, 'files read')
    df1 = dfscan1.sort_values("Patient",ascending = True, ignore_index=True)
#     print(df1)
    df2 = dfscan2.sort_values("Patient",ascending = True, ignore_index=True)
#     print(df2)
    df1cut = np.array(df1.iloc[:, 2:8])
    df2cut = np.array(df2.iloc[:, 2:8])
    dfpatients = df1.iloc[:, 0:2]
    dfdiff = pd.concat([dfpatients,pd.DataFrame(df2cut-df1cut)],axis=1,ignore_index=True)
                                          
    with pd.ExcelWriter(f'{dirName}/../volumedata_V1V2_{numcases}.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Visit 1')
        df2.to_excel(writer, sheet_name='Visit 2')
        dfdiff.to_excel(writer, sheet_name='V2-V1')

    subcutfat1 = df1["subcutfatvol"]
    subcutfat2 = df2["subcutfatvol"]
    intfat1 = df1["intfatvol"]
    intfat2 = df2["intfatvol"]
    psoas1 = df1["psoasvol"]
    psoas2 = df2["psoasvol"]
    muscle1 = df1["musclevol"]
    muscle2 = df2["musclevol"]
    totalfat1 = df1["totalfatvol"]
    totalfat2 = df2["totalfatvol"]
    totalmuscle1 = df1["totalmusclevol"]
    totalmuscle2 = df2["totalmusclevol"]
    zdim1 = df1["zdim"]
    zdim2 = df2["zdim"]
    fname = df1["Patient"]
    
    sfstats = repeatability_stats(subcutfat1, subcutfat2, calc_log=True)
    ifstats = repeatability_stats(intfat1, intfat2, calc_log=True)
#     pstats = repeatability_stats(psoas1, psoas2, calc_log=True)
    mstats = repeatability_stats(muscle1, muscle2, calc_log=True)
    tfstats = repeatability_stats(totalfat1, totalfat2, calc_log=True)
    tmstats = repeatability_stats(totalmuscle1, totalmuscle2, calc_log=True)

    savedata = {'TissueType':['SubcutFat','IntFat','Muscle','TotalFat','TotalMuscle'],
               'wCV': [sfstats['log']['sw_CoV'],ifstats['log']['sw_CoV'],mstats['log']['sw_CoV'],tfstats['log']['sw_CoV'],tmstats['log']['sw_CoV']],
               'ICC': [sfstats['log']['ICC'],ifstats['log']['ICC'],mstats['log']['ICC'],tfstats['log']['ICC'],tmstats['log']['ICC']],
               'LoA': [sfstats['log']['LoA'],ifstats['log']['LoA'],mstats['log']['LoA'],tfstats['log']['LoA'],tmstats['log']['LoA']]}

    data = {
    'Stats': ['sw', 'sw_CI',  'wCV (%)', 'r', 'r_CI','ICC', 'ICC_CI', 'LoA(%RCL, %RCU)', 'LoA_CI'],
    'SubcutFat(log)': [np.round(sfstats['log']['sw'],3),np.round(sfstats['log']['sw_CI'],3),np.round(sfstats['log']['sw_CoV'],3),\
                       np.round(sfstats['log']['r'],3),np.round(sfstats['log']['r_CI'],3),\
                       np.round(sfstats['log']['ICC'],3),np.round(sfstats['log']['ICC_CI'],3),\
                       np.round(sfstats['log']['LoA'],3),np.round(sfstats['log']['LoA_CI'],3)],
    'IntFat(log)': [np.round(ifstats['log']['sw'],3),np.round(ifstats['log']['sw_CI'],3),np.round(ifstats['log']['sw_CoV'],3),\
                       np.round(ifstats['log']['r'],3),np.round(ifstats['log']['r_CI'],3),\
                       np.round(ifstats['log']['ICC'],3),np.round(ifstats['log']['ICC_CI'],3),\
                       np.round(ifstats['log']['LoA'],3),np.round(ifstats['log']['LoA_CI'],3)],
#     'Psoas(log)': [np.round(pstats['log']['sw'],3),np.round(pstats['log']['sw_CI'],3),np.round(pstats['log']['sw_CoV'],3),\
#                        np.round(pstats['log']['r'],3),np.round(pstats['log']['r_CI'],3),\
#                        np.round(pstats['log']['ICC'],3),np.round(pstats['log']['ICC_CI'],3),\
#                        np.round(pstats['log']['LoA'],3),np.round(pstats['log']['LoA_CI'],3)],
    'Muscle(log)': [np.round(mstats['log']['sw'],3),np.round(mstats['log']['sw_CI'],3),np.round(mstats['log']['sw_CoV'],3),\
                       np.round(mstats['log']['r'],3),np.round(mstats['log']['r_CI'],3),\
                       np.round(mstats['log']['ICC'],3),np.round(mstats['log']['ICC_CI'],3),\
                       np.round(mstats['log']['LoA'],3),np.round(mstats['log']['LoA_CI'],3)],
    'TotalFat(log)': [np.round(tfstats['log']['sw'],3),np.round(tfstats['log']['sw_CI'],3),np.round(tfstats['log']['sw_CoV'],3),\
                       np.round(tfstats['log']['r'],3),np.round(tfstats['log']['r_CI'],3),\
                       np.round(tfstats['log']['ICC'],3),np.round(tfstats['log']['ICC_CI'],3),\
                       np.round(tfstats['log']['LoA'],3),np.round(tfstats['log']['LoA_CI'],3)],
    'TotalMuscle(log)': [np.round(tmstats['log']['sw'],3),np.round(tmstats['log']['sw_CI'],3),np.round(tmstats['log']['sw_CoV'],3),\
                       np.round(tmstats['log']['r'],3),np.round(tmstats['log']['r_CI'],3),\
                       np.round(tmstats['log']['ICC'],3),np.round(tmstats['log']['ICC_CI'],3),\
                       np.round(tmstats['log']['LoA'],3),np.round(tmstats['log']['LoA_CI'],3)]
    }
    dfstats = pd.DataFrame(data)
    dfsave = pd.DataFrame(savedata)
    print(exptfolder)
    display(dfstats)
    with pd.ExcelWriter(f'/Users/bthomas/Downloads/data_for_Belvin/midl24/repstats_{exptfolder}.xlsx') as writer:
        dfstats.to_excel(writer, sheet_name='Repeatability')
        dfsave.to_excel(writer, sheet_name='Savestats4Plot')
    
    f1, ax1 = pl.subplots(1,3) 
    plot_BA_in_axis(subcutfat1, subcutfat2, zdim1, zdim2, fname, ax1[0], parameter_name='Subcut Fat volume', units=r'AU')
    plot_BA_in_axis(intfat1, intfat2, zdim1, zdim2, fname, ax1[1], parameter_name='Int Fat Volume', units=r'AU')
    plot_BA_in_axis(totalfat1, totalfat2, zdim1, zdim2, fname, ax1[2], parameter_name='Total Fat volume', units=r'AU')
    pl.show()
    
    f2, ax2 = pl.subplots(1,2)
#     plot_BA_in_axis(psoas1, psoas2, zdim1, zdim2, fname, ax2[0], parameter_name='Psoas volume', units=r'AU')
    plot_BA_in_axis(muscle1, muscle2, zdim1, zdim2, fname, ax2[0], parameter_name='Muscle volume', units=r'AU')
    plot_BA_in_axis(totalmuscle1, totalmuscle2, zdim1, zdim2, fname, ax2[1], parameter_name='Total Muscle Volume', units=r'AU')
    pl.show()
    
#create Bland-Altman plot using statsmodels api 
#     import statsmodels.api as sm
    
#     f3, ax3 = pl.subplots(1,2)
#     ax3[0].set_title("Subcut fat volume")
#     sm.graphics.mean_diff_plot(subcutfat1, subcutfat2, ax = ax3[0])
#     sm.graphics.mean_diff_plot(intfat1, intfat2, ax = ax3[1])
#     ax3[1].set_title("Int fat volume")
#     pl.show()
    
#     f4, ax4 = pl.subplots(1,2)
#     ax4[0].set_title("Psoas volume")
#     sm.graphics.mean_diff_plot(psoas1, psoas2, ax = ax4[0])
#     ax4[1].set_title("Muscle volume")
#     sm.graphics.mean_diff_plot(muscle1, muscle2, ax = ax4[1])
#     pl.show()
    
#     f5, ax5 = pl.subplots(1,2)
#     ax5[0].set_title("Total Fat volume")
#     sm.graphics.mean_diff_plot(totalfat1, totalfat2, ax = ax5[0])
#     ax5[1].set_title("Total Muscle volume")
#     sm.graphics.mean_diff_plot(totalmuscle1, totalmuscle2, ax = ax5[1])
#     pl.show()

        
if __name__ == '__main__':
    main()