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
&emsp; -fmap correction using gre sequence only  
v2.1_noUnwarp: same as v2 but with no fmap correction  
v3.0: partial FOV preprocessing  
&emsp; -modified registration  
&emsp; -fmap correction using gre sequence  
v3.1:  
&emsp; -same as v3.0 but with no fmap correction  
**v4: macaque preprocessing: whole-brain  
&emsp; -fmap correction using gre sequence  
**v5.1: whole-brain preprocessing  
&emsp; -unwarping using SE-EPI and topup  
&emsp; -sigloss estimation using gre; mag and delta phase  
!v5.2: whole-brain preprocessing  
&emsp; -unwarping using SE-EPI and topup  
&emsp; -sigloss estimation using gre; mag and wrapped phase per echo  

** Under development  
! Most up-to-date using CBIC Prisma data  
