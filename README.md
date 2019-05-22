# fMRI Preprocess

Various scripts for preprocessing fMRI data.\

### Dependencies: 
afni\
fsl\
ants (for preprocess.py)\

### Python script
preprocess.py uses Nipype.\

### Bash scripts
The two *.sh scripts go together:\
preprocess_parameters.sh is for setting up the preprocessing parameters.\
preprocess.sh executes preprocessing functions.

To run:\
source preprocess_parameters.sh
bash preprocess.sh
