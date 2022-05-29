### Preprocessing_parameters (v1.3)
# Vince Man
# September 30, 2015


### Parameters to specify by the user for preprocessing ###

### Readme ###
### Set up the parameters for preprocessing here ###

# Define where the home folder is:
home=/projects/vincent/COGBD-project/Data

# Create a text file with all the group IDs
# Where is the group list? 
grplist=${home}/group_list.txt

# Create text files with all the subjects, one for each group
# Name them in the following format: <group>_subjectlist.txt

####### Parameters on standard images ######
standard_location=/opt/quarantine/FSL/5.0.7/build/data/standard
config_location=/opt/quarantine/FSL/5.0.7/build/etc/flirtsch

####### Parameters on structural data ######

# Define the filename for the anatomical data
anat_prefix=T1
brain_anat_prefix=T1_brain

###### Parameters on functional data ######

# Define the filename for the functional data
func_prefix=sprl_shft_al
brain_func_prefix=sprl_shft_al_brain

# TR of functional data
TR=2.0

# Number of disdaqs (initial volumes to delete)
num_disdaq=5

# Metric to use to estimate excessive motion outliers
motion_metric='dvars'
# (See http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLMotionOutliers for options)

# Apply slice time correction? 
slice_time_on=false # Enter true or false
acquis_interleaved='odd' 
reverse_slice_index='down'

# Coregistration options
coreg_DoF=12 # translation=3 rigid_body=6 global=7 traditional=9 affine=12
search_min=0
search_max=0

# Despike data? 
despike_on=true # Enter true or false

# WM and CSF covariate mask thresholds
WM_thresh=0.5
CSF_thresh=0.5

# Bandpass on? 
bandpass_on=true  # Enter true or false 
# Bandpass filter range
ftop=0.1 # Highest frequency in passband
fbot=0.01 # Lowest frequency in passband

# Smoothing on? 
smooth_on=true # Enter true or false
# Define smoothing kernel FWHM
fwhm=8
