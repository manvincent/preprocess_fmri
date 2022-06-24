# fMRI Preprocess

Various scripts for preprocessing fMRI data.

### Dependencies:
afni\
fsl\
ants (for preprocess.py)

### Python script
preprocess.py uses Nipype

v1: outdated (no fieldmap correction)  
v2.0: whole-brain preprocessing  
  * fmap correction using gre sequence only  *   
v2.1_noUnwarp: same as v2 but with no fmap correction  
v3.0: partial FOV preprocessing  
  * modified registration  *   
  * fmap correction using gre sequence  *
v3.1_noUnwarp:
  * same as v3_slab but with no fmap correction  *
**v4: macaque preprocessing: whole-brain
  * fmap correction using gre sequence  *
**v5.1: whole-brain preprocessing  
  * unwarping using SE-EPI and topup  *
  * sigloss estimation using gre; mag and delta phase  *
!v5.2: whole-brain preprocessing
  * unwarping using SE-EPI and topup  *
  * sigloss estimation using gre; mag and wrapped phase per echo  * 

** Under development  
! Most up-to-date using CBIC Prisma data  
