# fMRI Preprocess

Various scripts for preprocessing fMRI data.

### Dependencies: 
afni\
fsl\
ants (for preprocess.py)

### Python script
preprocess.py uses Nipype.  
v1: outdated (no fieldmap correction)  
v2: current whole-brain with fmap correction  
v2_noUnwarp: same as v2 but with no fmap correction  
v3: for partial FOV, modified registration, with fmap correction  

### Bash scripts
The two *.sh scripts go together:\
preprocess_parameters.sh is for setting up the preprocessing parameters.\
preprocess.sh executes preprocessing functions.

To run:\
source preprocess_parameters.sh  
bash preprocess.sh
