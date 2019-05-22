## Nipype preprocessing pipelin for data without fieldmap unwarpping 
## Vincent Man September 30, 2017

## Print the nipype graph to see details
## Needs to be formatted to conform to BIDS format, not ready yet
## Currently, the folder structure it expects is:
# Main Folder (expDir)
# --> Raw_nii (folder containing your raw data)
# --> --> Sub1 (subject folder)
# --> --> --> raw (folder containing all your raw images)
# --> --> --> --> run1.nii.gz
# --> --> --> --> run2.nii.gz
# --> --> --> --> run3.nii.gz ..etc (your EPIs for each run)
# --> --> --> --> t1.nii.gz (your structural) 
# --> --> Sub2 (subject folder)
## --> --> --> raw...etc


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
# Import prepackaged workflows
from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth


# Import modules for displaying plots remotely
import matplotlib
matplotlib.use('TKAgg')  # X11 back-end for Linux --> Mac OS
import pylab
from nilearn import plotting as niplt

### Experiment parameters
# Number of threads
num_cores = multiprocessing.cpu_count() # for whole preprocessing (here uses all)
ANTS_num_threads=1 # for the ANTS modules (can be different than num_cores)

# Locations
expDir = '/media/canlabadmin/ffd2/scratch2/vince/RF3B' # Location of your experiment (main) folder 
rawDir = expDir + os.sep + 'Raw_nii'  # Location of raw subject folders within your expDir
outDir = expDir + os.sep + 'Preprocessed'  # Make preprocessing directory within your expDir
if not os.path.exists(outDir):
    os.makedirs(outDir)
workDir = outDir + os.sep + 'WorkingDir'  # Working directory inside your 'Preprocessed' folder
if not os.path.exists(workDir):
    os.makedirs(workDir)
# Templates paths
templateDir = expDir + os.sep + 'Templates' # Folder of templates in your expDir 
priorDir = templateDir + os.sep + 'Oasis' + os.sep + 'Priors2'
# Standard space templates 
MNI_2mm_brain = templateDir + os.sep + 'MNI152_T1_2mm_brain.nii.gz'
MNI_2mm_brain_mask = templateDir + os.sep + 'MNI152_T1_2mm_brain_mask.nii.gz'
# ANTS templates for brain extraction
extractTemplate = templateDir +  os.sep + 'Oasis' + os.sep +'T_template0.nii.gz'
extractProb = templateDir +  os.sep + 'Oasis' + os.sep + 'T_template0_BrainCerebellumProbabilityMask.nii.gz'
extractMask = templateDir + os.sep + 'Oasis' + os.sep + 'T_template0_BrainCerebellumRegistrationMask.nii.gz'
# ANTS templates for segmentation
segCSF = priorDir + os.sep + 'priors1.nii.gz'
segGM = priorDir + os.sep + 'priors2.nii.gz'
segWM = priorDir + os.sep + 'priors3.nii.gz'


# Specify data to preprocess
# Count all subfolders
subList = next(os.walk(rawDir))[1]
runList = ['run1', 'run2', 'run3', 'run4', 'run5', 'run6'] # These should be the same name as your EPI Niftis
# Example debugging with single subject/run
#subList = ['RF3A_S07']
#runList = ['run1']

# Preprocessing parameters
TR = 1.0  # TR of data
numDisdaq = 4  # Number of initial volumes to delete
WMthresh=0.75 # WM covariate mask thresholds
CSFthresh=0.75 # CSF covariate mask thresholds
fwhmVal = 5 # Smoothing kernel in mm


#### End experiment parameters

# Brain extraction
# Specify custom (external) functions
def func_Extract(inEPI):
    import nibabel as nib
    import os
    from nilearn.masking import compute_epi_mask 
    niimg = nib.load(inEPI)
    niimg_extract = compute_epi_mask(niimg, 
        lower_cutoff=0.3, 
        upper_cutoff=0.8, 
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

# Brain extraction
anatExtract = pe.Node(ants.segmentation.BrainExtraction(
    dimension=3,
    brain_template=extractTemplate,
    brain_probability_mask=extractProb,
    extraction_registration_mask=extractMask,
    num_threads=ANTS_num_threads),
    name='anatExtract')

# Tissue segmentation (CSF, GM, WM)
# Select WM segmentation file from segmentation output
def get_wm(files):
    return files[-1]

# Select WM segmentation file from segmentation output
def get_csf(files):
    return files[0]

   
anatSegment = pe.Node(fsl.FAST(
    output_type='NIFTI_GZ',
    number_classes=3),
    name='anatSegment')

# anatSegment = pe.Node(ants.Atropos(
    # dimension=3,
    # initialization='PriorProbabilityImages',
    # prior_probability_images=[segCSF, segGM, segWM],
    # prior_probability_threshold=0.2,
    # prior_weighting=0.5,
    # number_of_tissue_classes=3,
    # likelihood_model='Gaussian',
    # n_iterations=5,
    # convergence_threshold=0.000001,
    # posterior_formulation='Socrates',
    # use_mixture_model_proportions=True,
    # save_posteriors=True

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

# Pick the middle volume from the given run
def middleVol(func):
    from nibabel import load
    funcfile = func
    _,_,_,timepoints = load(funcfile).get_shape()
    return int((timepoints/2)-1)

extractEPIref = pe.Node(fsl.ExtractROI(
    t_size=1,
    output_type='NIFTI_GZ'),
    name = 'extractEPIref')

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
    transforms=['Affine'],
    transform_parameters=[(0.1,)],
    metric=['MI'],
    metric_weight=[1.0],
    radius_or_number_of_bins=[32],
    number_of_iterations=[[1000,500,250,100]],
    convergence_threshold=[1e-06],
    convergence_window_size=[20],
    shrink_factors=[[8,4,2,1]],
    smoothing_sigmas=[[3.0,2.0,1.0,0.0]],
    write_composite_transform = True,
    num_threads=ANTS_num_threads),
    name='epi2anat')

# Normalise anatomical to standard space
anat2std = pe.Node(ants.Registration(
    fixed_image=MNI_2mm_brain,
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
    smoothing_sigmas=[[4, 2, 1],[4,2,1],[1, 0.5, 0]],
    sigma_units=['vox','vox','vox'],
    write_composite_transform = True,
    num_threads=ANTS_num_threads),
    name='anat2std')


# Merge registration (epi2anat) and normalisation (anat2std)
merge = pe.Node(util.Merge(2), 
    infields=['in1','in2'],
    name='merge')

# Split funcitonal by volumes before apply transformations
funcSplit = pe.Node(fsl.Split(
    dimension='t',
    output_type='NIFTI_GZ'), 
    name='funcSplit')

# Apply transformations
applyTransFunc = pe.MapNode(ants.ApplyTransforms(
    args='--float',
    reference_image=MNI_2mm_brain,
    dimension=3,
    interpolation = 'BSpline',
    invert_transform_flags = [False, False],
    terminal_output = 'file'),
    iterfield=['input_image'],
    name='applyTransFunc')

# Merge back all volumes
funcMerge = pe.Node(fsl.utils.Merge(
    dimension='t',
    tr=TR,
    output_type='NIFTI_GZ'), 
    name='funcMerge')

# Covariate prep
# Compute inverse (t1 --> epi) transfomration 
applyTransInvCSF = pe.Node(ants.ApplyTransforms(
    args='--float',
    dimension=3,
    interpolation = 'BSpline',
    invert_transform_flags = [True],
    terminal_output = 'file'),
    name='applyTransInvCSF')

applyTransInvWM = pe.Node(ants.ApplyTransforms(
    args='--float',
    dimension=3,
    interpolation = 'BSpline',
    invert_transform_flags = [True],
    terminal_output = 'file'),
    name='applyTransInvWM')

createWMmask = pe.Node(fsl.Threshold(
    thresh=WMthresh,
    output_type='NIFTI_GZ'),
    name='createWMmask')

createCSFmask = pe.Node(fsl.Threshold(
    thresh=CSFthresh,
    output_type='NIFTI_GZ'),
    name='createCSFmask')

WMmeanTS = pe.Node(fsl.ImageMeants(),
    name='WMmeanTS')

CSFmeanTS = pe.Node(fsl.ImageMeants(),
    name='CSFmeanTS')

# Create regressor file (concatenate regressors to make a matrix)
def create_Regressor(motionMatrix,outlierMatrix,CSFinput,WMinput):
    import pandas as pd
    import os
    df_motion = []
    df_outlier = []
    df_CSF = []
    df_WM = []
    df_concat = []    
    df_motion = pd.read_table(motionMatrix,sep='  ',engine='python')
    df_outlier = pd.read_table(outlierMatrix,sep='   ',engine='python')
    df_CSF = pd.read_table(CSFinput)
    df_WM = pd.read_table(WMinput)
    df_concat = pd.concat([df_motion,df_outlier,df_CSF,df_WM], ignore_index=False, axis=1)
    df_concat.to_csv('covarOut.txt',sep=str('\t'),header=True,index=False)
    return os.path.abspath('covarOut.txt')

createRegressor = pe.Node(Function(
    input_names=['motionMatrix','outlierMatrix','CSFinput','WMinput'],
    output_names=['out_file'],
    function=create_Regressor),
    name='createRegressor')


# Smoothing workflow (SUSAN)
smoothWF  = create_susan_smooth() 
smoothWF.inputs.inputnode.fwhm = fwhmVal
smoothWF.inputs.inputnode.mask_file=MNI_2mm_brain_mask
    
def smoothOut(susanList):
    return susanList[0]

# Mask functional image with extracted brain mask
finalMask = pe.Node(fsl.maths.ApplyMask(
    mask_file=MNI_2mm_brain_mask,
    output_type='NIFTI_GZ'),
    name='finalMask')

### Workflow connections ### 
### Anatomical sub workflow ###
anatWF = pe.Workflow(name='anatWF')
anatWF.base_dir = os.path.join(workDir)

# inputnode - a function free node to iterate over the list of subject names
inputNode_anat = pe.Node(util.IdentityInterface(
    fields=['subject_id']),
    name='inputNode_anat')

# SelectFiles for Input
inputData_anat = {'anat': rawDir + os.sep + '{subject_id}/Raw/t1.nii.gz'}

selectFiles_anat = pe.Node(nio.SelectFiles(
    inputData_anat,
    base_directory=expDir),
    name="selectFiles_anat")

outputNode_anat = pe.Node(util.IdentityInterface(
    fields=['anatWM','anatCSF','extractBrain']),
    name='outputNode_anat')

### Registration workflow ###
regWF = pe.Workflow(name='regWF')
regWF.base_dir = os.path.join(workDir)

inputNode_reg = pe.Node(util.IdentityInterface(
    fields=['extractBrain','in_EPI','EPI_refVol']),
    name='inputNode_reg')

outputNode_reg = pe.Node(util.IdentityInterface(
    fields=['trans_EPI','epi2anat_xfm']), 
    name='outputNode_reg')

### Covariate workflow ###
covarWF = pe.Workflow(name='covarWF')
covarWF.base_dir = os.path.join(workDir)

inputNode_covar = pe.Node(util.IdentityInterface(
    fields=['in_CSF','in_WM','in_EPI','EPI_refVol','in_motion','in_outlier','in_transform']),
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
inputData = {'func': rawDir + os.sep + '{subject_id}/Raw/{session_id}.nii.gz'}
    
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
    ('highres001_BrainExtractionBrain.nii.gz','t1_brain.nii.gz'),
    ('highres001_BrainExtractionBrain_pve_0.nii.gz','t1_brain_CSF.nii.gz'),
    ('highres001_BrainExtractionBrain_pve_1.nii.gz','t1_brain_GM.nii.gz'),
    ('highres001_BrainExtractionBrain_pve_2.nii.gz','t1_brain_WM.nii.gz'),
    ('_masked.nii.gz','_funcMask.nii.gz'),
    ('vol0000_trans_merged.nii.gz','func_noDespike_noSmooth.nii.gz'),
    ('vol0000_trans_merged_despike_thresh.nii.gz','prep_noSmooth.nii.gz'),
    ('vol0000_trans_merged_despike_thresh_smooth_funcMask.nii.gz','prep_Smooth.nii.gz'),
    ('_smooth0/','')
    ]
dataSink.inputs.substitutions = substitutions

# Anat workflow connections
anatWF.connect([
    (inputNode_anat, selectFiles_anat, [('subject_id','subject_id')]),
    (selectFiles_anat, anatExtract, [('anat','anatomical_image')]),
    # Segmentation of anatomical image 
    (anatExtract,anatSegment,[('BrainExtractionBrain','in_files')]),
    # Send to output node
    (anatExtract, outputNode_anat, [('BrainExtractionBrain','extractBrain')]),
    (anatSegment, outputNode_anat, [(('partial_volume_files',get_csf),'anatCSF'),
        (('partial_volume_files',get_wm),'anatWM')]),
    # Outputs to datasink
    (anatExtract, dataSink, [('BrainExtractionBrain','Extras.@extractBrain')]),
    (anatSegment, dataSink, [('partial_volume_files','Extras.@segmentBrain')]),
    ])

# Registration/Normalisation workflow connections
regWF.connect([
    (inputNode_reg, epi2anat, [('EPI_refVol','moving_image')]),
    (inputNode_reg, epi2anat, [('extractBrain','fixed_image')]),
    (inputNode_reg, anat2std, [('extractBrain','moving_image')]),
    (epi2anat, merge, [('composite_transform','in2')]),    
    (anat2std, merge, [('composite_transform','in1')]),
    # Split functional image by volumes
    (inputNode_reg, funcSplit, [('in_EPI','in_file')]),
    # Send to transformation application 
    (merge, applyTransFunc, [('out','transforms')]),
    (funcSplit , applyTransFunc, [('out_files','input_image')]),
    # Merge functional volumes back to image
    (applyTransFunc, funcMerge, [('output_image','in_files')]),
    (funcMerge,outputNode_reg,[('merged_file','trans_EPI')]),
    (epi2anat, outputNode_reg, [('composite_transform','epi2anat_xfm')]),
    # Outputs to dataSink
    (epi2anat, dataSink, [('composite_transform','Extras.@epi2anat_xfm')]),
    (anat2std, dataSink, [('composite_transform','Extras.@anat2std_xfm')]),
    (outputNode_reg, dataSink, [('trans_EPI','Extras.@normalisedBrain')]),
    ])

# Covar workflow connections
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
    # Concatenate all covariates to make matrix
    (inputNode_covar, createRegressor, [('in_motion','motionMatrix'), ('in_outlier','outlierMatrix')]),    
    (WMmeanTS, createRegressor, [('out_file','WMinput')]),
    (CSFmeanTS, createRegressor, [('out_file','CSFinput')]),        
    ])


# Main workflow connections
preproc.connect([
    (infoSource, selectFiles_preproc, [('subject_id', 'subject_id'), ('session_id','session_id')]),
    (infoSource, anatWF, [('subject_id','inputNode_anat.subject_id')]),
    (selectFiles_preproc, funcExtract, [('func', 'inEPI')]),
    (funcExtract, funcMask, [('outEPI','mask_file')]),
    (selectFiles_preproc, funcMask, [('func','in_file')]),
    (funcMask, volTrim, [('out_file','in_file')]),
    (volTrim, motionOutliers, [('roi_file','in_file')]),
    (volTrim, motionCorrect, [('roi_file','in_file')]),
    (motionCorrect, extractEPIref, [(('out_file',middleVol),'t_min'), ('out_file','in_file')]),    
    # Registration and normalisation                
    (motionCorrect, regWF, [('out_file','inputNode_reg.in_EPI')]),
    (extractEPIref, regWF, [('roi_file','inputNode_reg.EPI_refVol')]),
    (anatWF, regWF, [('outputNode_anat.extractBrain','inputNode_reg.extractBrain')]),
    # Connections to covariate regression workflow
    (motionCorrect, covarWF, [('out_file','inputNode_covar.in_EPI'),
        ('par_file','inputNode_covar.in_motion')]),
    (anatWF, covarWF, [('outputNode_anat.anatCSF','inputNode_covar.in_CSF'), ('outputNode_anat.anatWM','inputNode_covar.in_WM')]),    
    (regWF, covarWF, [('outputNode_reg.epi2anat_xfm','inputNode_covar.in_transform')]),
    (extractEPIref, covarWF, [('roi_file','inputNode_covar.EPI_refVol')]),
    (motionOutliers, covarWF, [('out_file','inputNode_covar.in_outlier')]),
    # Send for despiking    
    (regWF, despike, [('outputNode_reg.trans_EPI','in_file')]),
    (despike, posLimit, [('out_file','in_file')]),
    # Send for smoothing
    (posLimit, smoothWF, [('out_file','inputnode.in_files')]),
    # Apply final mask
    (smoothWF, finalMask, [(('outputnode.smoothed_files',smoothOut),'in_file')]),
    # Send final files to output node
    (covarWF, outputNode, [('createRegressor.out_file','covariateDesign')]),
    (finalMask, outputNode, [('out_file','cleanedFile')]),
    # Outputs to to dataSink
    (funcMask, dataSink, [('out_file','Extras.@warped_func')]),
    (posLimit, dataSink, [('out_file','Extras.@unsmoothed')]), # unsmoothed func
    (outputNode, dataSink, [('covariateDesign','Preprocessed.@covariates'),
        ('cleanedFile','Preprocessed.@cleanedSmoothed')]), # Final smoothed func
    ])

# Visualize overall workflow
preproc.write_graph(graph2use='colored', simple_form=True)

# Run!
preproc.run('MultiProc', plugin_args={'n_procs': num_cores})
