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
v3_slab: for partial FOV, modified registration, with fmap correction  
v3_slab_noUnwarp: same as v3_slab but with no fmap correction  
**v4: macaque preprocessing: whole-brain with 3D fmap correction

** Upcoming
