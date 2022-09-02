#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 7 2022

Nipype preprocessing pipeline for macaque fMRI data

Input data must be converted to nifti and correctly oriented

Designed for whole-brain POV
Uses 3D GRE fieldmaps for unwarping

Print the nipype graph to see details

Not yet formatted for BIDS format
Expected folder structure:
Main Folder (expDir)
--> Raw_nii (folder containing your raw data)
--> --> {sessID}_func.nii.gz
--> --> {sessID}_b0_mag.nii.gz (3d fieldmap magnitude)
--> --> {sessID}_b0_phase.nii.gz (3d fieldmap phase)
--> --> t1.nii.gz (your structural)

@author: Vincent Man vman@caltech.edu
"""

# Import global engines
import os  # system
from os.path import abspath
import multiprocessing
import nibabel as nb
import numpy as np
# Import numpy engines
import nipype.pipeline.engine as pe  # the workflow and node wrappers
from nipype import Function
import nipype.interfaces.io as nio  # Input/Output
import nipype.interfaces.utility as util  # utility
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.afni as afni  # afni
import nipype.interfaces.ants as ants  # ANTS
import nipype.algorithms.confounds as conf


# Import modules for displaying plots remotely
import matplotlib
matplotlib.use('TKAgg')  # X11 back-end for Linux --> Mac OS
import pylab

### Experiment parameters
# Number of threads
num_cores = multiprocessing.cpu_count() # for whole preprocessing (here uses all)
ANTS_num_threads = 32  # for the ANTS modules (can be different than num_cores)

# Locations
expDir = '/state/partition1/home/vman/rewmod/svenData/Data/monkey_raw' # Location of your experiment (main) folder
outDir = f'{expDir}/Preprocessed_fez' # Location of your experiment (main) folder
if not os.path.exists(outDir):
    os.makedirs(outDir)
workDir = f'{outDir}/WorkingDir'  # Working directory inside your 'Preprocessed' folder
if not os.path.exists(workDir):
    os.makedirs(workDir)
# Templates paths
# Specify whether to transform to 'native' or 'standard' (NMT or D99) space
template = 'native'
# Template images
templateDir = f'{expDir}/templates/D99_v2.0_dist/'
standard_brain = f'{templateDir}/D99_template.nii.gz'
standard_brain_mask = f'{templateDir}/D99_template_mask.nii.gz'
standard_atlas = f'{templateDir}/D99_atlas_v2.0.nii.gz'

# Specify transformation flags
if template == 'standard':
    inv_xfm_flag = [False, False]
elif template == 'native':
    inv_xfm_flag = [False]
else:
    raise ValueError('Error! Template incorrectly specified')

# Specify data to preprocess
# Count all subfolders
subList = ['fez']
runList = ['141028','141029','141107','150319','150323','150325']

# Preprocessing parameters
TR = 1.04  # TR of data
numDisdaq = 0  # Number of initial volumes to delete
WMthresh=0.9 # WM covariate mask cthresholds
CSFthresh=0.9 # CSF covariate mask thresholds
aCompCor_varThresh = 0.5
tCompCor_varThresh = 0.5
smoothing_kernels = [5,6,8]

# Fieldmap parameters
# From fieldmap
deltaTE = 2.46
# From EPI
echoSpacing = 0.000524993
TE = 0.027
unwarpDir = 'z-'
# Reference resample resolution
resampleRes = 1 # in mm, isotropic

#### End experiment parameters

# Functional Brain extraction
# Bias correction on the functional
extractEPIref1 = pe.Node(fsl.ExtractROI(
    t_size=1,
    output_type='NIFTI_GZ'),
    name = 'extractEPIref1')

funcBiasCorrect = pe.Node(ants.N4BiasFieldCorrection(
    dimension = 3,
    n_iterations = [50,50,30],
    convergence_threshold = 0.0,
    shrink_factor = 4,
    bspline_fitting_distance = 50,
    save_bias = True,
    num_threads=ANTS_num_threads),
    name = 'funcBiasCorrect')

applyFuncBiasCorrect = pe.Node(fsl.maths.BinaryMaths(
    operation = 'div',
    nan2zeros=False,
    output_type='NIFTI_GZ'),
    name='applyFuncBiasCorrect')

def func_Extract(inEPI):
    import nibabel as nib
    import os
    from nilearn.masking import compute_epi_mask
    niimg = nib.load(inEPI)
    niimg_extract = compute_epi_mask(niimg,
        lower_cutoff=0.2,
        upper_cutoff=0.76,
        connected=True,
        opening=2,
        exclude_zeros=False,
        ensure_finite=True,
        verbose=0)
    nib.save(niimg_extract, 'func_brain.nii.gz')
    return os.path.abspath('func_brain.nii.gz')

funcExtract = pe.Node(Function(
    input_names=['inEPI'],
    output_names=['outEPI'],
    function=func_Extract),
    name="funcExtract")


# Mask functional image with extracted brain mask
funcMask = pe.Node(fsl.maths.ApplyMask(
    output_type='NIFTI_GZ'),
    name='funcMask')

# Bias correction on the anatomical
anatBiasCorrect = pe.Node(ants.N4BiasFieldCorrection(
    dimension = 3,
    n_iterations = [50,50,30,20],
    convergence_threshold = 0.0,
    shrink_factor = 3,
    bspline_fitting_distance = 300,
    num_threads=ANTS_num_threads),
    name = 'anatBiasCorrect')

# Brain extraction
anatExtract = pe.Node(fsl.BET(
    robust = True,
    mask = True,
    frac = 0.65),
    name = 'anatExtract')

# Remove initial volumes
volTrim = pe.Node(fsl.ExtractROI(
    t_min=numDisdaq,
    t_size=-1,
    output_type='NIFTI_GZ'),
    name='volTrim')

# Estimate extreme motion outliers
motionOutliers = pe.Node(fsl.MotionOutliers(),
    name="motionOutliers")

# Estimate motion parameters
motionCorrect = pe.Node(fsl.MCFLIRT(
    cost='normcorr',
    dof=12,
    mean_vol=True,
    save_plots=True,
    save_rms=True,
    output_type='NIFTI_GZ'),
    name="motionCorrect")

# Pick the middle volume from the given run
def middleVol(func):
    from nibabel import load
    funcfile = func
    _,_,_,timepoints = load(funcfile).shape
    return int((timepoints/2)-1)

extractEPIref2 = pe.Node(fsl.ExtractROI(
    t_size=1,
    output_type='NIFTI_GZ'),
    name = 'extractEPIref2')

# Despike raw data
despike = pe.Node(afni.preprocess.Despike(
    outputtype='NIFTI_GZ'),
    name="despike")

# Remove negative values (from despike procedure)
posLimit = pe.Node(fsl.maths.Threshold(
    nan2zeros=True,
    thresh=0,
    output_type='NIFTI_GZ'),
    name='posLimit')

smooth = pe.Node(fsl.Smooth(),
    iterables = ('fwhm',smoothing_kernels),
    name='smooth')


## Fieldmap workflow
# Prepare fieldmap
magVolTrim = pe.Node(fsl.ExtractROI(
    t_min=0,
    t_size=1,
    output_type='NIFTI_GZ'),
    name='magVolTrim')

magBiasCorrect = pe.Node(ants.N4BiasFieldCorrection(
    dimension = 3,
    n_iterations = [50,50,30,20],
    convergence_threshold = 0.0,
    shrink_factor = 3,
    bspline_fitting_distance = 300,
    num_threads=ANTS_num_threads),
    name = 'magBiasCorrect')

magExtract = pe.Node(fsl.BET(
    robust = True,
    frac = 0.4,
    vertical_gradient = -0.10),
    name = 'magExtract')

erode = pe.Node(fsl.maths.ErodeImage(
    output_type = 'NIFTI_GZ'),
    name = 'erode')

prepareFmap = pe.Node(fsl.PrepareFieldmap(
    delta_TE = deltaTE,
    scanner = 'SIEMENS'),
    name = 'prepareFmap')

filterFmap = pe.Node(fsl.preprocess.FUGUE(
    median_2dfilter = True,
    save_fmap = True),
    name = 'filterFmap')

epiMean = pe.Node(fsl.maths.MeanImage(
    dimension = 'T',
    output_type = 'NIFTI_GZ'),
    name = 'epiMean')

## Register fmap image to magnitude image
epi2mag = pe.Node(fsl.FLIRT(
    dof = 6,
    cost = 'normcorr'),
    name = 'epi2mag')

convertXFM = pe.Node(fsl.ConvertXFM(
    invert_xfm = True),
    name = 'convertXFM')

fmap2epi = pe.Node(fsl.FLIRT(
    apply_xfm = True),
    name = 'fmap2epi')

## Fieldmap unwarping
epiUnwarp = pe.Node(fsl.preprocess.FUGUE(
    dwell_time = echoSpacing,
    unwarp_direction = unwarpDir,
    forward_warping = False,
    nokspace = True),
    name = 'epiUnwarp')

# Estimate signal loss map
sigloss = pe.Node(fsl.SigLoss(
    echo_time = TE),
    name = 'sigloss')


# Registration/Normalisation workflow
# Register functional to anatomical
epi2anat = pe.Node(ants.Registration(
    dimension=3,
    float=False,
    output_transform_prefix='epi2anat_',
    interpolation='Linear',
    winsorize_lower_quantile=0.005,
    winsorize_upper_quantile=0.995,
    use_histogram_matching=False,
    initial_moving_transform_com=1,
    transforms=['Rigid','Affine'],
    transform_parameters=[(0.1,),(0.1,)],
    metric=['Mattes','Mattes'],
    metric_weight=[1.0, 1.0],
    radius_or_number_of_bins=[32,32],
    number_of_iterations=[[1000,500,250,100],[1000,500,250,100]],
    convergence_threshold=[1e-06,1e-06],
    convergence_window_size=[20,20],
    shrink_factors=[[8,4,2,1],[8,4,2,1]],
    smoothing_sigmas=[[3.0,2.0,1.0,0.0],[3.0,2.0,1.0,0.0]],
    write_composite_transform = False,
    num_threads=ANTS_num_threads),
    name='epi2anat')

# Normalise anatomical to standard space
anat2std = pe.Node(ants.Registration(
    dimension=3,
    float=False,
    output_transform_prefix='anat2std_',
    output_warped_image='anat2std_warp.nii.gz',
    interpolation='Linear',
    winsorize_lower_quantile=0.005,
    winsorize_upper_quantile=0.995,
    use_histogram_matching=[False, False, True], # Set to true for epiAlign
    initial_moving_transform_com=1,
    transforms=['Rigid','Affine','SyN'],
    transform_parameters=[(0.1,),(0.1,), (0.1, 3.0, 0.0)],
    metric=['Mattes','Mattes','CC'],
    metric_weight=[1.0, 1.0, 1.0],
    radius_or_number_of_bins=[32,32,4],
    sampling_strategy=['Regular','Regular',None],
    sampling_percentage=[0.3,0.3,None],
    number_of_iterations=[[10000, 11110, 11110],[10000, 11110, 11110],[100, 30, 20]],
    convergence_threshold= [1.e-8,1.e-8,-0.01],
    convergence_window_size=[20,20,5],
    use_estimate_learning_rate_once = [True,True,True],
    shrink_factors=[[3,2,1],[3,2,1],[4,2,1]],
    smoothing_sigmas=[[4,2,1],[4,2,1],[1, 0.5, 0]],
    sigma_units=['vox','vox','vox'],
    write_composite_transform = False,
    num_threads=ANTS_num_threads),
    name='anat2std')

# Reverse the order of the transformation results
def reverseList(inList):
    inList.reverse()
    return inList

# Merge registration (epi2anat) and normalisation (anat2std)
merge = pe.Node(util.Merge(2),
    infields=['in1','in2'],
    name='merge')

# Split funcitonal by volumes before applying transformations
funcSplit = pe.Node(fsl.Split(
    dimension='t',
    output_type='NIFTI_GZ'),
    name='funcSplit')

# Resample the reference image to the resolution of the functional
resampleStd = pe.Node(fsl.FLIRT(
    apply_isoxfm = resampleRes,
    no_resample_blur = True),
    name='resampleStd')


resampleAnat = pe.Node(fsl.FLIRT(
    apply_isoxfm = resampleRes,
    no_resample_blur = True),
    name='resampleAnat')

# Threshold and mask the resample reference image
threshRef =  pe.Node(fsl.Threshold(
    thresh=10,
    output_type='NIFTI_GZ'),
    name='threshRef')

maskRef =  pe.Node(fsl.UnaryMaths(
    operation='bin',
    output_type='NIFTI_GZ'),
    name='maskRef')



# Transform and mask atlas
applyTransInvAtlas = pe.Node(ants.ApplyTransforms(
    args='--float',
    dimension=3,
    interpolation = 'Linear',
    invert_transform_flags = [True, False],
    num_threads=ANTS_num_threads),
    iterfield=['input_image'],
    name='applyTransInvAtlas')

# Transform and mask signal loss map
applyTransFunc_sigloss = pe.Node(ants.ApplyTransforms(
    args='--float',
    dimension=3,
    interpolation = 'BSpline',
    invert_transform_flags = inv_xfm_flag,
    num_threads=ANTS_num_threads),
    iterfield=['input_image'],
    name='applyTransFunc_sigloss')

# Mask sigloss image with extracted brain mask
siglossMask = pe.Node(fsl.maths.ApplyMask(
    output_type='NIFTI_GZ'),
    name='siglossMask')

# Apply transformations to functional image
applyTransFunc = pe.MapNode(ants.ApplyTransforms(
    args='--float',
    dimension=3,
    interpolation = 'BSpline',
    invert_transform_flags = inv_xfm_flag,
    num_threads=ANTS_num_threads),
    iterfield=['input_image'],
    name='applyTransFunc')

# Merge back all volumes
def func_Merge(in_files):
    import nibabel as nib
    import os
    from nibabel.funcs import concat_images
    inputs = [nib.load(img) for img in in_files]
    merged = concat_images(images=inputs)
    nib.save(merged, 'func_merged.nii.gz')
    return os.path.abspath('func_merged.nii.gz')

funcMerge = pe.Node(Function(
    input_names=['in_files'],
    output_names=['merged_file'],
    function=func_Merge),
    name="funcMerge")

# Tissue segmentation (CSF, GM, WM)
# Select WM segmentation file from segmentation output
def get_wm(files):
    return files[-1]

# Select WM segmentation file from segmentation output
def get_csf(files):
    return files[0]

stdSegment = pe.Node(fsl.FAST(
    output_type='NIFTI_GZ',
    number_classes=3),
    name='stdSegment')

# Covariate prep
# Compute inverse (t1 --> epi) transformation
applyTransInvCSF = pe.Node(ants.ApplyTransforms(
    args='--float',
    dimension=3,
    interpolation = 'BSpline',
    invert_transform_flags = [True, True, False]),
    name='applyTransInvCSF')

applyTransInvWM = pe.Node(ants.ApplyTransforms(
    args='--float',
    dimension=3,
    interpolation = 'BSpline',
    invert_transform_flags = [True, True, False]),
    name='applyTransInvWM')

createWMmask = pe.Node(fsl.Threshold(
    thresh=WMthresh,
    output_type='NIFTI_GZ'),
    name='createWMmask')

createCSFmask = pe.Node(fsl.Threshold(
    thresh=CSFthresh,
    output_type='NIFTI_GZ'),
    name='createCSFmask')

merge_compCor = pe.Node(util.Merge(3),
    infields=['in1','in2','in3'],
    name='merge_compCor')


aCompCor = pe.Node(conf.ACompCor(
    merge_method = 'union',
    variance_threshold = aCompCor_varThresh,
    repetition_time = TR,
    pre_filter = 'cosine',
    failure_mode = 'NaN'),
    name = 'aCompCor')

tCompCor = pe.Node(conf.TCompCor(
    merge_method = 'union',
    variance_threshold = tCompCor_varThresh,
    repetition_time = TR,
    pre_filter = 'cosine',
    percentile_threshold=0.05,
    failure_mode = 'NaN'),
    name = 'tCompCor')

WMmeanTS = pe.Node(fsl.ImageMeants(),
    name='WMmeanTS')

CSFmeanTS = pe.Node(fsl.ImageMeants(),
    name='CSFmeanTS')

# Create regressor file (concatenate regressors to make a matrix)
def create_Regressor(motionMatrix,outlierMatrix,CSFinput,WMinput,aCompCor_in,tCompCor_in):
    import pandas as pd
    import os
    df_motion = pd.read_table(motionMatrix, header=None,sep='  ',engine='python')
    df_motion.columns = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    df_CSF = pd.read_table(CSFinput, header=None)
    df_CSF.columns = ['CSF']
    df_WM = pd.read_table(WMinput, header=None)
    df_WM.columns = ['WM']
    df_aCompCor = pd.read_table(aCompCor_in,sep='\t',header=[0],engine='python')
    df_tCompCor = pd.read_table(tCompCor_in,sep='\t',header=[0],engine='python')
    df_outlier = pd.read_table(outlierMatrix,header=None,sep='   ',engine='python')
    df_concat = pd.concat([df_motion,
                           df_CSF,
                           df_WM,
                           df_aCompCor,
                           df_tCompCor,
                           df_outlier], ignore_index=False, axis=1)
    df_concat.to_csv('covarOut.txt',sep=str('\t'),header=True,index=False)
    return os.path.abspath('covarOut.txt')

createRegressor = pe.Node(Function(
    input_names=['motionMatrix','outlierMatrix','CSFinput','WMinput','aCompCor_in','tCompCor_in'],
    output_names=['out_file'],
    function=create_Regressor),
    name='createRegressor')


### Workflow connections ###
### Fieldmap unwarping workflow ###
fmapWF = pe.Workflow(name='fmapWF')
fmapWF.base_dir = os.path.join(workDir)

# inputnode - a function free node to iterate over the list of subject names
inputNode_fmap = pe.Node(util.IdentityInterface(
    fields=['subject_id','session_id','in_EPI']),
    name='inputNode_fmap')

inputData_fmap = {'mag': f'{expDir}/{{subject_id}}/Raw_nii/{{session_id}}_b0_mag.nii.gz',
                  'phase': f'{expDir}/{{subject_id}}/Raw_nii/{{session_id}}_b0_phase.nii.gz'}

selectFiles_fmap = pe.Node(nio.SelectFiles(
    inputData_fmap,
    base_directory=expDir),
    name="selectFiles_fmap")

outputNode_fmap = pe.Node(util.IdentityInterface(
    fields=['unwarpedBrain','out_sigloss']),
    name='outputNode_fmap')

### Anatomical sub workflow ###
anatWF = pe.Workflow(name='anatWF')
anatWF.base_dir = os.path.join(workDir)

# inputnode - a function free node to iterate over the list of subject names
inputNode_anat = pe.Node(util.IdentityInterface(
    fields=['subject_id']),
    name='inputNode_anat')

# SelectFiles for Input
inputData_anat = {'anat': f'{expDir}/{{subject_id}}/Raw_nii/t1.nii.gz'}

selectFiles_anat = pe.Node(nio.SelectFiles(
    inputData_anat,
    base_directory=expDir),
    name="selectFiles_anat")

outputNode_anat = pe.Node(util.IdentityInterface(
    fields=['extractBrain','anatMask']),
    name='outputNode_anat')

### Registration workflow ###
regWF = pe.Workflow(name='regWF')
regWF.base_dir = os.path.join(workDir)

inputNode_reg = pe.Node(util.IdentityInterface(
    fields=['extractBrain','in_EPI','EPI_refVol','in_sigloss']),
    name='inputNode_reg')

inputData_reg = {'standard': standard_brain,
                 'atlas': standard_atlas}

selectFiles_reg = pe.Node(nio.SelectFiles(
    inputData_reg,
    base_directory=expDir),
    name="selectFiles_reg")

outputNode_reg = pe.Node(util.IdentityInterface(
    fields=['trans_EPI','stdMask','merge_xfm','stdCSF','stdWM']),
    name='outputNode_reg')

### Covariate workflow ###
covarWF = pe.Workflow(name='covarWF')
covarWF.base_dir = os.path.join(workDir)

inputNode_covar = pe.Node(util.IdentityInterface(
    fields=['in_CSF','in_WM','in_EPI','EPI_refVol','in_motion','in_outlier','in_transform','epiMask']),
    name='inputNode_covar')

### (MAIN) Functional Workflow ###
preproc = pe.Workflow(name='preprocess')
preproc.base_dir = os.path.join(workDir)

# Infosource - a function free node to iterate over the list of subject names
infoSource = pe.Node(util.IdentityInterface(
    fields=['subject_id','session_id']),
    name="infoSource")

infoSource.iterables = [
    ('subject_id', subList),
    ('session_id', runList)]

# SelectFiles for Input
inputData = {'func': f'{expDir}/{{subject_id}}/Raw_nii/{{session_id}}_func.nii.gz'}

selectFiles_preproc = pe.Node(nio.SelectFiles(
    inputData,
    base_directory=expDir),
    name="selectFiles_preproc")

outputNode = pe.Node(util.IdentityInterface(
    fields=['cleanedFile','covariateDesign']),
    name='outputNode')

# DataSink for Output
dataSink = pe.Node(nio.DataSink(
    base_directory=outDir),
    name="dataSink")

# Create 'container' files -- separate output files for each subID
preproc.connect([(infoSource, dataSink, [('subject_id','container')]),
    ])

# Use the following DataSink output substitutions
substitutions = [
    ('_subject_id', ''),
    ('_session_id_', ''),
    ('D99_template_pve_0.nii.gz','std_CSF_mask.nii.gz'),
    ('D99_template_pve_1.nii.gz','std_GM_mask.nii.gz'),
    ('D99_template_pve_2.nii.gz','std_WM_mask.nii.gz'),
    ('D99_atlas_v2.0_trans.nii.gz','std_atlas_trans.nii.gz'),
    ('t1_corrected_brain.nii.gz','t1_brain.nii.gz'),
    ('t1_corrected_brain_mask.nii.gz','t1_brain_mask.nii.gz'),
    ('t1_corrected_brain_flirt_thresh_bin.nii.gz','t1_brain_trans_mask.nii.gz'),
    ('t1_corrected_brain_flirt.nii.gz','t1_brain_trans.nii.gz'),
    ('func_merged.nii.gz','func_noDespike_noSmooth.nii.gz'),
    ('func_merged_despike_thresh.nii.gz','prep_noSmooth.nii.gz'),
    ('func_merged_despike_thresh_smooth.nii.gz','prep_Smooth.nii.gz'),
    ('D99_template_flirt.nii.gz','std_trans.nii.gz'),
    ('D99_template_flirt_thresh_bin.nii.gz','std_mask_trans.nii.gz'),
    ('D99_template_flirt_pve_0.nii.gz','std_CSF_mask_trans.nii.gz'),
    ('D99_template_flirt_pve_1.nii.gz','std_GM_mask_trans.nii.gz'),
    ('D99_template_flirt_pve_2.nii.gz','std_WM_mask_trans.nii.gz'),
    ]

dataSink.inputs.substitutions = substitutions

# Fieldmap workflow connnections
fmapWF.connect([
    (inputNode_fmap, selectFiles_fmap, [('subject_id','subject_id'),('session_id','session_id')]),
    (selectFiles_fmap, magVolTrim, [('mag','in_file')]),
    (magVolTrim, magBiasCorrect, [('roi_file','input_image')]),
    # Prepare fieldmap
    (magBiasCorrect, magExtract, [('output_image','in_file')]),
    (magExtract, erode, [('out_file','in_file')]),
    (erode, prepareFmap, [('out_file','in_magnitude')]),
    (selectFiles_fmap, prepareFmap, [('phase','in_phase')]),
    (prepareFmap, filterFmap, [('out_fieldmap','fmap_in_file')]),
    # Extimate registration matrix between EPI and magnitude
    (inputNode_fmap, epiMean, [('in_EPI','in_file')]),
    (erode, epi2mag, [('out_file','reference')]),
    (epiMean, epi2mag, [('out_file','in_file')]),
    # Register fmap to EPI
    (epi2mag, convertXFM, [('out_matrix_file','in_file')]),
    (convertXFM, fmap2epi, [('out_file','in_matrix_file')]),
    (filterFmap, fmap2epi, [('fmap_out_file','in_file')]),
    (epiMean, fmap2epi, [('out_file','reference')]),
    # Estimate signal loss map
    (fmap2epi, sigloss, [('out_file','in_file')]),
    # Fieldmap unwarping
    (fmap2epi, epiUnwarp, [('out_file','fmap_in_file')]),
    (inputNode_fmap, epiUnwarp, [('in_EPI','in_file')]),
    # Send to output node
    (epiUnwarp, outputNode_fmap, [('unwarped_file','unwarpedBrain')]),
    (sigloss, outputNode_fmap, [('out_file','out_sigloss')]),
    ])

# Anat workflow connections
anatWF.connect([
    (inputNode_anat, selectFiles_anat, [('subject_id','subject_id')]),
    # Bias correction
    (selectFiles_anat, anatBiasCorrect,  [('anat','input_image')]),
    # Brain extraction and segmentation
    (anatBiasCorrect, anatExtract, [('output_image','in_file')]),
    # Send to output node
    (anatExtract, outputNode_anat, [('out_file','extractBrain'),
                                    ('mask_file','anatMask')]),
    # Outputs to datasink
    (anatExtract, dataSink, [('out_file','Extras.@extractBrain'),
                             ('mask_file','Extras.@anatMask')]),
    ])

# Registration/Normalisation workflow connections
if template == 'native':
    regWF.connect([
        # Resample final reference image to resolution of functional
        (inputNode_reg, resampleAnat, [('extractBrain','reference'),
                                      ('extractBrain','in_file')]),
        # Create mask from resample reference image
        (resampleAnat , threshRef, [('out_file','in_file')]),
        (threshRef , maskRef, [('out_file','in_file')]),
        # Estimate transformations
        (inputNode_reg, epi2anat, [('EPI_refVol','moving_image')]),
        (resampleAnat , epi2anat, [('out_file','fixed_image')]),
        (resampleAnat , anat2std, [('out_file','moving_image')]),
        (selectFiles_reg, anat2std, [('standard','fixed_image')]),
        # Merge epi2anat and anat2std estimated transformation matrices for covariance
        (epi2anat, merge, [('reverse_transforms','in1')]),
        (anat2std, merge, [('reverse_transforms','in2')]),
        # Split functional image by volumes
        (inputNode_reg, funcSplit, [('in_EPI','in_file')]),
        # Send to transformation application
        (epi2anat, applyTransFunc, [(('forward_transforms',reverseList),'transforms')]),
        (funcSplit , applyTransFunc, [('out_files','input_image')]),
        (resampleAnat , applyTransFunc, [('out_file','reference_image')]),
        # Merge functional volumes back to image
        (applyTransFunc, funcMerge, [('output_image','in_files')]),
        (funcMerge, outputNode_reg,[('merged_file','trans_EPI')]),
        (merge, outputNode_reg, [('out','merge_xfm')]),
        # Get segmented standard space image
        (selectFiles_reg, stdSegment, [('standard','in_files')]),
        (stdSegment, outputNode_reg, [(('partial_volume_files',get_csf),'stdCSF'),
                                        (('partial_volume_files',get_wm),'stdWM')]),
        # Transform atlas into anatomical space
        (anat2std, applyTransInvAtlas, [('reverse_transforms','transforms')]),
        (selectFiles_reg, applyTransInvAtlas, [('atlas','input_image')]),
        (resampleAnat , applyTransInvAtlas, [('out_file','reference_image')]),
        # Transform signal loss map
        (epi2anat, applyTransFunc_sigloss, [(('forward_transforms',reverseList),'transforms')]),
        (inputNode_reg, applyTransFunc_sigloss, [('in_sigloss','input_image')]),
        (resampleAnat , applyTransFunc_sigloss, [('out_file','reference_image')]),
        (applyTransFunc_sigloss , siglossMask, [('output_image','in_file')]),
        (maskRef, siglossMask, [('out_file','mask_file')]),
        # Outputs to dataSink
        (resampleAnat , dataSink, [('out_file','Extras.@resampleAnat')]),
        (maskRef , dataSink, [('out_file','Extras.@resampleAnat_mask')]),
        (stdSegment, dataSink, [('partial_volume_files','Extras.@segmentBrain')]),
        (siglossMask, dataSink, [('out_file','Extras.@trans_sigloss')]),
        (applyTransInvAtlas, dataSink, [('output_image','Extras.@trans_atlas')]),
        (outputNode_reg, dataSink, [('trans_EPI','Extras.@trans_noDespike')]),
        ])
elif template == 'standard':
    regWF.connect([
        # Resample final reference image to resolution of functional
        (selectFiles_reg, resampleStd, [('standard','reference'),
                                      ('standard','in_file')]),
        # Create mask from resample reference image
        (resampleStd , threshRef, [('out_file','in_file')]),
        (threshRef , maskRef, [('out_file','in_file')]),
        (maskRef , outputNode_reg, [('out_file','stdMask')]),
        # Estimate transformations
        (inputNode_reg, epi2anat, [('EPI_refVol','moving_image')]),
        (inputNode_reg , epi2anat, [('extractBrain','fixed_image')]),
        (inputNode_reg , anat2std, [('extractBrain','moving_image')]),
        (resampleStd, anat2std, [('out_file','fixed_image')]),
        # Merge epi2anat and anat2std estimated transformation matrices
        (epi2anat, merge, [(('forward_transforms',reverseList),'in2')]),
        (anat2std, merge, [(('forward_transforms',reverseList),'in1')]),
        # Split functional image by volumes
        (inputNode_reg, funcSplit, [('in_EPI','in_file')]),
        # Send to transformation application
        (merge, applyTransFunc, [('out','transforms')]),
        (funcSplit , applyTransFunc, [('out_files','input_image')]),
        (resampleStd , applyTransFunc, [('out_file','reference_image')]),
        # Merge functional volumes back to image
        (applyTransFunc, funcMerge, [('output_image','in_files')]),
        (funcMerge, outputNode_reg,[('merged_file','trans_EPI')]),
        # Get segmented standard space image
        (resampleStd, stdSegment, [('out_file','in_files')]),
        (stdSegment, outputNode_reg, [(('partial_volume_files',get_csf),'stdCSF'),
                                        (('partial_volume_files',get_wm),'stdWM')]),
        # Transform signal loss map
        (merge, applyTransFunc_sigloss, [('out','transforms')]),
        (inputNode_reg, applyTransFunc_sigloss, [('in_sigloss','input_image')]),
        (resampleStd , applyTransFunc_sigloss, [('out_file','reference_image')]),
        (applyTransFunc_sigloss , siglossMask, [('output_image','in_file')]),
        (maskRef, siglossMask, [('out_file','mask_file')]),
        # Outputs to dataSink
        (resampleStd , dataSink, [('out_file','Extras.@resampleStd')]),
        (maskRef , dataSink, [('out_file','Extras.@resampleStd_mask')]),
        (stdSegment, dataSink, [('partial_volume_files','Extras.@segmentBrain')]),
        (siglossMask, dataSink, [('out_file','Extras.@trans_sigloss')]),
        (outputNode_reg, dataSink, [('trans_EPI','Extras.@trans_noDespike')]),
        ])

# Covar workflow connections
if template == 'native':
    covarWF.connect([
        # Inverse transform (t1 --> epi space)
        (inputNode_covar, applyTransInvCSF, [('in_transform','transforms'),
                                             ('in_CSF','input_image'),
                                             ('EPI_refVol','reference_image')]),
        (inputNode_covar, applyTransInvWM, [('in_transform','transforms'),
                                            ('in_WM','input_image'),
                                            ('EPI_refVol','reference_image')]),
        # Create CSF and WM masks
        (applyTransInvCSF, createCSFmask, [('output_image','in_file')]),
        (applyTransInvWM, createWMmask, [('output_image','in_file')]),
        # Calculate mean TS for CSF and WM
        (inputNode_covar, CSFmeanTS, [('in_EPI','in_file')]),
        (createCSFmask, CSFmeanTS, [('out_file','mask')]),
        (inputNode_covar, WMmeanTS, [('in_EPI','in_file')]),
        (createWMmask, WMmeanTS, [('out_file','mask')]),
        # Calculate compCor
        # aCompCor
        (inputNode_covar, aCompCor, [('in_EPI','realigned_file')]),
        (createCSFmask, merge_compCor, [('out_file','in1')]),
        (createWMmask, merge_compCor, [('out_file','in2')]),
        (inputNode_covar, merge_compCor, [('epiMask','in3')]),
        (merge_compCor, aCompCor, [('out','mask_files')]),
        # tCompCor
        (inputNode_covar, tCompCor, [('in_EPI','realigned_file')]),
        (inputNode_covar, tCompCor, [('epiMask','mask_files')]),
        # Concatenate all covariates to make matrix
        (inputNode_covar, createRegressor, [('in_motion','motionMatrix'),
                                            ('in_outlier','outlierMatrix')]),
        (WMmeanTS, createRegressor, [('out_file','WMinput')]),
        (CSFmeanTS, createRegressor, [('out_file','CSFinput')]),
        (aCompCor, createRegressor, [('components_file','aCompCor_in')]),
        (tCompCor, createRegressor, [('components_file','tCompCor_in')]),
        ])
elif template == 'standard':
    covarWF.connect([
        # Create CSF and WM masks
        (inputNode_covar, createCSFmask, [('in_CSF','in_file')]),
        (inputNode_covar, createWMmask, [('in_WM','in_file')]),
        # Calculate mean TS for CSF and WM
        (inputNode_covar, CSFmeanTS, [('in_EPI','in_file')]),
        (createCSFmask, CSFmeanTS, [('out_file','mask')]),
        (inputNode_covar, WMmeanTS, [('in_EPI','in_file')]),
        (createWMmask, WMmeanTS, [('out_file','mask')]),
        # Calculate compCor
        # aCompCor
        (inputNode_covar, aCompCor, [('in_EPI','realigned_file')]),
        (createCSFmask, merge_compCor, [('out_file','in1')]),
        (createWMmask, merge_compCor, [('out_file','in2')]),
        (inputNode_covar, merge_compCor, [('epiMask','in3')]),
        (merge_compCor, aCompCor, [('out','mask_files')]),
        # tCompCor
        (inputNode_covar, tCompCor, [('in_EPI','realigned_file')]),
        (inputNode_covar, tCompCor, [('epiMask','mask_files')]),
        # Concatenate all covariates to make matrix
        (inputNode_covar, createRegressor, [('in_motion','motionMatrix'),
                                            ('in_outlier','outlierMatrix')]),
        (WMmeanTS, createRegressor, [('out_file','WMinput')]),
        (CSFmeanTS, createRegressor, [('out_file','CSFinput')]),
        (aCompCor, createRegressor, [('components_file','aCompCor_in')]),
        (tCompCor, createRegressor, [('components_file','tCompCor_in')]),
        ])

# Main workflow connections
if template == 'native':
    preproc.connect([
        (infoSource, selectFiles_preproc, [('subject_id', 'subject_id'), ('session_id','session_id')]),
        (infoSource, anatWF, [('subject_id','inputNode_anat.subject_id')]),
        (infoSource, fmapWF, [('subject_id','inputNode_fmap.subject_id'),('session_id','inputNode_fmap.session_id')]),
        # Estimate bias field and functional brain extraction / bias correction
        (selectFiles_preproc, extractEPIref1, [(('func',middleVol), 't_min'),
                                                ('func','in_file')]),
        (extractEPIref1, funcBiasCorrect, [('roi_file', 'input_image')]),
        (selectFiles_preproc, applyFuncBiasCorrect, [('func','in_file')]),
        (funcBiasCorrect, applyFuncBiasCorrect, [('bias_image','operand_file')]),
        (applyFuncBiasCorrect, funcExtract, [('out_file', 'inEPI')]),
        (funcExtract, funcMask, [('outEPI','mask_file')]),
        (applyFuncBiasCorrect, funcMask, [('out_file','in_file')]),
        # Trim and estimate motion
        (funcMask, volTrim, [('out_file','in_file')]),
        (volTrim, motionOutliers, [('roi_file','in_file')]),
        (volTrim, motionCorrect, [('roi_file','in_file')]),
        # Fieldmap workflow
        (motionCorrect, fmapWF, [('out_file','inputNode_fmap.in_EPI')]),
        (fmapWF, extractEPIref2, [(('outputNode_fmap.unwarpedBrain',middleVol),'t_min'),
                                  ('outputNode_fmap.unwarpedBrain','in_file')]),
        # Registration and normalisation
        (fmapWF, regWF, [('outputNode_fmap.unwarpedBrain','inputNode_reg.in_EPI'),
                         ('outputNode_fmap.out_sigloss','inputNode_reg.in_sigloss')]),
        (extractEPIref2, regWF, [('roi_file','inputNode_reg.EPI_refVol')]),
        (anatWF, regWF, [('outputNode_anat.extractBrain','inputNode_reg.extractBrain')]),
        # Connections to covariate regression workflow
        (funcExtract, covarWF, [('outEPI','inputNode_covar.epiMask')]),
        (fmapWF, covarWF, [('outputNode_fmap.unwarpedBrain','inputNode_covar.in_EPI')]),
        (motionCorrect, covarWF, [('par_file','inputNode_covar.in_motion')]),
        (regWF, covarWF, [('outputNode_reg.merge_xfm','inputNode_covar.in_transform'),
                          ('outputNode_reg.stdCSF','inputNode_covar.in_CSF'),
                          ('outputNode_reg.stdWM','inputNode_covar.in_WM')]),
        (extractEPIref2, covarWF, [('roi_file','inputNode_covar.EPI_refVol')]),
        (motionOutliers, covarWF, [('out_file','inputNode_covar.in_outlier')]),
        # Send for despiking
        (regWF, despike, [('outputNode_reg.trans_EPI','in_file')]),
        (despike, posLimit, [('out_file','in_file')]),
        (posLimit, smooth, [('out_file','in_file')]),
        # Send final files to output node
        (covarWF, outputNode, [('createRegressor.out_file','covariateDesign')]),
        (posLimit, outputNode, [('out_file','cleanedFile')]),
        # Outputs to to dataSink
        (smooth, dataSink, [('smoothed_file','Extras.@cleanedSmoothed_')]),
        (outputNode, dataSink, [('covariateDesign','Preprocessed.@covariates'),
            ('cleanedFile','Preprocessed.@cleanedSmoothed')]), # Final smoothed func
        ])
elif template == 'standard':
    preproc.connect([
        (infoSource, selectFiles_preproc, [('subject_id', 'subject_id'), ('session_id','session_id')]),
        (infoSource, anatWF, [('subject_id','inputNode_anat.subject_id')]),
        (infoSource, fmapWF, [('subject_id','inputNode_fmap.subject_id'),('session_id','inputNode_fmap.session_id')]),
        # Estimate bias field and functional brain extraction / bias correction
        (selectFiles_preproc, extractEPIref1, [(('func',middleVol), 't_min'),
                                                ('func','in_file')]),
        (extractEPIref1, funcBiasCorrect, [('roi_file', 'input_image')]),
        (selectFiles_preproc, applyFuncBiasCorrect, [('func','in_file')]),
        (funcBiasCorrect, applyFuncBiasCorrect, [('bias_image','operand_file')]),
        (applyFuncBiasCorrect, funcExtract, [('out_file', 'inEPI')]),
        (funcExtract, funcMask, [('outEPI','mask_file')]),
        (applyFuncBiasCorrect, funcMask, [('out_file','in_file')]),
        # Trim and estimate motion
        (funcMask, volTrim, [('out_file','in_file')]),
        (volTrim, motionOutliers, [('roi_file','in_file')]),
        (volTrim, motionCorrect, [('roi_file','in_file')]),
        # Fieldmap workflow
        (motionCorrect, fmapWF, [('out_file','inputNode_fmap.in_EPI')]),
        (fmapWF, extractEPIref2, [(('outputNode_fmap.unwarpedBrain',middleVol),'t_min'),
                                  ('outputNode_fmap.unwarpedBrain','in_file')]),
        # Registration and normalisation
        (fmapWF, regWF, [('outputNode_fmap.unwarpedBrain','inputNode_reg.in_EPI'),
                         ('outputNode_fmap.out_sigloss','inputNode_reg.in_sigloss')]),
        (extractEPIref2, regWF, [('roi_file','inputNode_reg.EPI_refVol')]),
        (anatWF, regWF, [('outputNode_anat.extractBrain','inputNode_reg.extractBrain')]),
        # Connections to covariate regression workflow
        (regWF, covarWF, [('outputNode_reg.stdMask','inputNode_covar.epiMask'),
                          ('outputNode_reg.trans_EPI','inputNode_covar.in_EPI'),
                          ('outputNode_reg.stdCSF','inputNode_covar.in_CSF'),
                          ('outputNode_reg.stdWM','inputNode_covar.in_WM')]),
        (motionCorrect, covarWF, [('par_file','inputNode_covar.in_motion')]),
        (motionOutliers, covarWF, [('out_file','inputNode_covar.in_outlier')]),
        # Send for despiking
        (regWF, despike, [('outputNode_reg.trans_EPI','in_file')]),
        (despike, posLimit, [('out_file','in_file')]),
        (posLimit, smooth, [('out_file','in_file')]),
        # Send final files to output node
        (covarWF, outputNode, [('createRegressor.out_file','covariateDesign')]),
        (posLimit, outputNode, [('out_file','cleanedFile')]),
        # Outputs to to dataSink
        (smooth, dataSink, [('smoothed_file','Extras.@cleanedSmoothed_')]),
        (outputNode, dataSink, [('covariateDesign','Preprocessed.@covariates'),
                                ('cleanedFile','Preprocessed.@cleanedSmoothed')]), # Final smoothed func
        ])


# Visualize overall workflow
preproc.write_graph(graph2use='colored', simple_form=False)

# Run!
preproc.run('MultiProc', plugin_args={'n_procs': num_cores})
