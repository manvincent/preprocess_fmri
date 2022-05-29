### Preprocessing v1.3
# Vince Man
# September 30, 2015

# Uses FSL (5.0.7) and AFNI (12.16.2014)

### Readme ###
# You need to run preprocess_parameters.sh before running this script

# Organize your data according to this structure: home/group/subject/
	# The 'home' folder is your study folder
	# The 'group' folder is your group folder - e.g. Controls folder vs. Schizophrenia folder
		# If you only have one group in your study, just create a stand-in "Controls" folder 
	# Each subject should have their own 'subject' folder (nested within their respective groups)
# Inside each subject's folder, create a "Raw" folder and put the raw data (T1, epi) inside! 
# Example:
	# my_study
		# -> HC
			# -> HC_sub1
				# -> Raw
					# - T1.nii.gz
					# - epi.nii.gz
			# -> HC_sub2
			# -> HC_sub3 ...etc
		# -> Schizophrenia 	
			# -> SCZ_sub1
			# -> SCZ_sub2
			# -> SCZ_sub3 ...etc

# Create a text file with all the group IDs

# Create text files with all the subjects, one for each group
	# Name them in the following format: <group>_subjectlist.txt
	# The subject IDs in the text files should be the same as the subject-specific folder names! 

# Put both the group list and subject list in the home (e.g. my_study) level folder
##############

echo 'Get ready for your data to be preprocessed!'
echo 'Did you specify preprocess_parameters.sh?'

# Load required programs (depends on system)
module load FSL/5.0.7
module load AFNI/2014.12.16

for group in $(cat ${grplist}); do 
export curr_sublist=${home}/${group}_subjectlist_FIX.txt
for sub in $(cat ${curr_sublist}); do

# Specify directory variables
raw_dir=${home}/${group}/${sub}/Raw
preprocess_dir=${home}/${group}/${sub}/Preprocessed

#  Create a directory for preprocessed data
if [ ! -d ${preprocess_dir} ]; then
mkdir ${preprocess_dir}
fi
# Create a directory to hold extra regressors (i.e. covariates)
if [ ! -d ${preprocess_dir}/extra_regressors ]; then 
mkdir ${preprocess_dir}/extra_regressors
fi
extra_regressors_dir=${preprocess_dir}/extra_regressors

# Brain-extract anatomical
if [ ! -d ${preprocess_dir}/a_Brain_extract ]; then
mkdir ${preprocess_dir}/a_Brain_extract
fi
brain_extract_dir=${preprocess_dir}/a_Brain_extract

if [ ! -e ${preprocess_dir}/a_Brain_extract/${brain_anat_prefix}.nii* ]; then
echo 'Extracting the anatomical brain for' ${sub} 
3dSkullStrip \
-input ${raw_dir}/${anat_prefix}.nii \
-prefix ${brain_extract_dir}/${brain_anat_prefix} \
-orig_vol
cd ${brain_extract_dir}
3dAFNItoNIFTI -prefix ${brain_anat_prefix}.nii.gz ${brain_anat_prefix}+tlrc
fslreorient2std ${brain_extract_dir}/${brain_anat_prefix}.nii.gz ${brain_extract_dir}/${brain_anat_prefix}_reoriented.nii.gz
mv ${brain_extract_dir}/${brain_anat_prefix}_reoriented.nii.gz ${brain_extract_dir}/${brain_anat_prefix}.nii.gz
cd ${preprocess_dir}
fi

# Perform FAST segmentation on anatomical
if [ ! -d ${preprocess_dir}/b_Segmentation ]; then
mkdir ${preprocess_dir}/b_Segmentation
fi
segmentation_dir=${preprocess_dir}/b_Segmentation

if [ ! -e ${preprocess_dir}/b_Segmentation/${brain_anat_prefix}_wmseg.nii* ]; then
echo 'Performing segmentation on anatomical brain for' ${sub}
fast \
-t 1 -g --nopve \
-o ${segmentation_dir}/${brain_anat_prefix} \
${brain_extract_dir}/${brain_anat_prefix}

# Output: 0=CSF; 1=GM; 2=WM 
# Rename outputs: 
mv ${segmentation_dir}/${brain_anat_prefix}_seg_0.nii.gz ${segmentation_dir}/${brain_anat_prefix}_csfseg.nii.gz
mv ${segmentation_dir}/${brain_anat_prefix}_seg_1.nii.gz ${segmentation_dir}/${brain_anat_prefix}_gmseg.nii.gz
mv ${segmentation_dir}/${brain_anat_prefix}_seg_2.nii.gz ${segmentation_dir}/${brain_anat_prefix}_wmseg.nii.gz
fi

# Brain-extract functional
if [ ! -d ${preprocess_dir}/a_Brain_extract ]; then
mkdir ${preprocess_dir}/a_Brain_extract
fi

if [ ! -e ${preprocess_dir}/a_Brain_extract/${brain_func_prefix}.nii* ]; then
echo 'Extracting the functional brain for' ${sub} 
bet \
${raw_dir}/${func_prefix} \
${brain_extract_dir}/${brain_func_prefix} \
-R -F 
fslreorient2std ${brain_extract_dir}/${brain_func_prefix}.nii.gz ${brain_extract_dir}/${brain_func_prefix}_reoriented.nii.gz
mv ${brain_extract_dir}/${brain_func_prefix}_reoriented.nii.gz ${brain_extract_dir}/${brain_func_prefix}.nii.gz
gzip -d -f ${brain_extract_dir}/${brain_func_prefix}.nii.gz 

echo 'Re-aligning functional image to T1 origin for' ${sub}
@Align_Centers -base ${brain_extract_dir}/${brain_anat_prefix}.nii.gz -dset ${brain_extract_dir}/${brain_func_prefix}.nii -cm 
fi

# Delete initial functional volumes
if [ ! -d ${preprocess_dir}/1_Disdaq ]; then
mkdir ${preprocess_dir}/1_Disdaq
fi
disdaq_dir=${preprocess_dir}/1_Disdaq

if [ ! -e ${preprocess_dir}/1_Disdaq/${brain_func_prefix}_disdaq.nii* ]; then
echo 'Removing' ${num_disdaq} 'initial volumes for' ${sub}
3dTcat \
-prefix ${brain_func_prefix}_disdaq \
-session ${disdaq_dir} \
-tr ${TR} \
${brain_extract_dir}/${brain_func_prefix}_shft.nii[${num_disdaq}..$]

cd ${disdaq_dir}
3dAFNItoNIFTI -prefix ${brain_func_prefix}_disdaq.nii.gz ${brain_func_prefix}_disdaq+tlrc
cd ${preprocess_dir}
fi

# Estimate excessive head motion 
if [ ! -d ${extra_regressors_dir}/motion_outliers ]; then
mkdir ${extra_regressors_dir}/motion_outliers
fi
motion_outliers_dir=${extra_regressors_dir}/motion_outliers

if [ ! -e ${extra_regressors_dir}/motion_outliers/${sub}_motion_confound* ]; then
echo 'Estimating outlier motion confounds for' ${sub}
fsl_motion_outliers \
-i ${disdaq_dir}/${brain_func_prefix}_disdaq.nii.gz \
-o ${motion_outliers_dir}/${sub}_motion_confound \
-p ${motion_outliers_dir}/${sub}_metric_plot \
-s ${motion_outliers_dir}/${sub}_${motion_metric}_metric \
--dummy=0 --${motion_metric}  
fi

# Slice time correction [opt]
if [ ${slice_time_on} = true ]; then

if [ ! -d ${preprocess_dir}/Opt_i_Slicetime ]; then
mkdir ${preprocess_dir}/Opt_i_Slicetime
fi
slicetime_dir=${preprocess_dir}/Opt_i_Slicetime

if [ ! -e ${slicetime_dir}/*st* ]; then
echo 'Slice-time correction for' ${sub}
slicetimer \
-i ${disdaq_dir}/${brain_func_prefix}_disdaq \
-o ${slicetime_dir}/${brain_func_prefix}_disdaq_st \
--${acquis_interleaved} \
--${reverse_slice_index} 

data_loc=${slicetime_dir}
data=${brain_func_prefix}_disdaq_st
fi
elif [ ${slice_time_on} =  false ]; then
data_loc=${disdaq_dir}
data=${brain_func_prefix}_disdaq
fi

# Estimate motion parameters
if [ ! -d ${extra_regressors_dir}/motion_parameters ]; then
mkdir ${extra_regressors_dir}/motion_parameters
fi
motion_parameters_dir=${extra_regressors_dir}/motion_parameters

if [ ! -d ${preprocess_dir}/2_MotionFiltered ]; then
mkdir ${preprocess_dir}/2_MotionFiltered
fi
motionfiltered_dir=${preprocess_dir}/2_MotionFiltered

if [ ! -e ${motionfiltered_dir}/*mcf.nii* ]; then
echo 'Estimating motion parameters for' ${sub}
mcflirt -in ${data_loc}/${data} -mats -plots
# Clean up
mv ${data_loc}/*mcf.mat ${motion_parameters_dir}
mv ${data_loc}/*mcf.par ${motion_parameters_dir}
mv ${data_loc}/${data}_mcf.nii.gz ${motionfiltered_dir}
fi
data_loc=${motionfiltered_dir}
data=${data}_mcf

## Estimate linear coregistration parameters: func to structural 

# Make Registration Folders
if [ ! -d ${preprocess_dir}/3_Registration ]; then
mkdir ${preprocess_dir}/3_Registration
fi 
registration_dir=${preprocess_dir}/3_Registration

if [ ! -d ${registration_dir}/func2highres ]; then
mkdir ${registration_dir}/func2highres
fi
func2highres_dir=${registration_dir}/func2highres

if [ ! -e ${func2highres_dir}/${sub}_flirt_func2highres.mat ]; then
echo 'Extracting first volume of functional in preparation for co-registration for' ${sub}
fslroi \
${data_loc}/${data} \
${func2highres_dir}/${data}_1stvol \
0 1 
echo 'Estimating co-registration parameters for' ${sub}
flirt -in ${func2highres_dir}/${data}_1stvol \
-ref ${brain_extract_dir}/${brain_anat_prefix} \
-omat ${func2highres_dir}/${sub}_flirt_func2highres.mat \
-out ${func2highres_dir}/${sub}_flirt_func2highres \
-dof ${coreg_DoF} \
-searchrx ${search_min} ${search_max} \
-searchry ${search_min} ${search_max} \
-searchrz ${search_min} ${search_max}

mv ${func2highres_dir}/${sub}_flirt_func2highres.nii.gz ${func2highres_dir}/${sub}_flirt_func2highres_QC.nii.gz
fi

## Estimate linear coregistration parameters: structural to standard
if [ ! -d ${registration_dir}/highres2std ]; then
mkdir ${registration_dir}/highres2std
fi
highres2std_dir=${registration_dir}/highres2std

if [ ! -e ${registration_dir}/highres2std/${sub}_fnirt_coefficients* ]; then
echo 'Estimating normalisation parameters for' ${sub}

# Initial flirt of structural to standard to get -aff output from FNIRT
echo 'Computing initial linear affine matrix with FLIRT for' ${sub}
flirt -in ${brain_extract_dir}/${brain_anat_prefix} \
-ref ${standard_location}/MNI152_T1_2mm_brain  \
-omat ${highres2std_dir}/${sub}_flirt_affine.mat \
-dof ${coreg_DoF} \
-searchrx ${search_min} ${search_max} \
-searchry ${search_min} ${search_max} \
-searchrz ${search_min} ${search_max}

# FNIRT
echo 'Computing nonlinear parameters with FNIRT for' ${sub}
fnirt --ref=${standard_location}/MNI152_T1_2mm_brain \
--in=${brain_extract_dir}/${brain_anat_prefix} \
--config=${config_location}/T1_2_MNI152_2mm \
--aff=${highres2std_dir}/${sub}_flirt_affine.mat \
--cout=${highres2std_dir}/${sub}_fnirt_coefficients 
fi

## Despike data [opt]
if [ ${despike_on} = true ]; then

if [ ! -d ${preprocess_dir}/Opt_ii_Despike ]; then
mkdir ${preprocess_dir}/Opt_ii_Despike
fi
Despike_dir=${preprocess_dir}/Opt_ii_Despike

if [ ! -e ${Despike_dir}/despiked* ]; then
gzip -f -d ${data_loc}/${data}.nii.gz
echo 'Despiking for' ${sub}
cd ${Despike_dir}
3dDespike \
-ignore 0 \
-nomask \
-NEW \
-prefix despiked_${data} \
${data_loc}/${data}.nii

3dAFNItoNIFTI -prefix despiked_${data}.nii.gz despiked_${data}+tlrc
cd ${preprocess_dir}
fi
data_loc=${Despike_dir}
data=despiked_${data}

elif [ ${despike_on} =  false ]; then
data_loc=${data_loc}
data=${data}
fi

## Extract WM and CSF time series to use as covariates in GLM
if [ ! -d ${extra_regressors_dir}/brain_covar ]; then
mkdir ${extra_regressors_dir}/brain_covar
fi
brain_covar_dir=${extra_regressors_dir}/brain_covar

if [ ! -e ${func2highres_dir}/${sub}_flirt_highres2func.mat ]; then
# Move segmented structural images to functional space by inversing estimated transformation matrix
echo 'Inversing estimated func2highres transformation matrix for' ${sub} 
convert_xfm \
-omat ${func2highres_dir}/${sub}_flirt_highres2func.mat \
-inverse \
${func2highres_dir}/${sub}_flirt_func2highres.mat
fi

if [ ! -e ${segmentation_dir}/${sub}_WM_funcspace_mask* ]; then
# Register structural data to functional space for covariate timeseries estimation!
echo 'Register WM mask to functional space for covariate timeseries estimation, processing' ${sub}
flirt \
-in ${segmentation_dir}/${brain_anat_prefix}_wmseg \
-ref ${data_loc}/${data} \
-applyxfm \
-init ${func2highres_dir}/${sub}_flirt_highres2func.mat \
-out ${segmentation_dir}/${sub}_WM_funcspace

echo 'Thresholding WM mask to exclude values below' ${WM_thresh} 'for' ${sub}
fslmaths ${segmentation_dir}/${sub}_WM_funcspace -thr ${WM_thresh}  ${segmentation_dir}/${sub}_WM_funcspace_mask
fi

if [ ! -e ${segmentation_dir}/${sub}_CSF_funcspace_mask* ]; then
echo 'Register CSF mask to functional space for covariate timeseries estimation, processing' ${sub}
flirt \
-in ${segmentation_dir}/${brain_anat_prefix}_csfseg \
-ref ${data_loc}/${data} \
-applyxfm \
-init ${func2highres_dir}/${sub}_flirt_highres2func.mat \
-out ${segmentation_dir}/${sub}_CSF_funcspace

echo 'Thresholding CSF mask to exclude values below' ${CSF_thresh} 'for' ${sub}
fslmaths ${segmentation_dir}/${sub}_CSF_funcspace -thr ${CSF_thresh}  ${segmentation_dir}/${sub}_CSF_funcspace_mask
fi

if [ ! -e ${brain_covar_dir}/${sub}_WMconfound_meants.txt ]; then 
# Extracting mean timeseries for WM and CSF
echo 'Extracting mean TS of WM for'  ${sub}
fslmeants \
-i ${data_loc}/${data} \
-m ${segmentation_dir}/${sub}_WM_funcspace_mask \
-o ${brain_covar_dir}/${sub}_WMconfound_meants.txt \
-w

echo 'Extracting mean TS of CSF for' ${sub}
fslmeants \
-i ${data_loc}/${data} \
-m ${segmentation_dir}/${sub}_CSF_funcspace_mask \
-o ${brain_covar_dir}/${sub}_CSFconfound_meants.txt \
-w
fi

## Regress out motion parameters and motion outliers 
if [ ! -d ${preprocess_dir}/4_GLMresiduals ]; then
mkdir ${preprocess_dir}/4_GLMresiduals
fi
GLMresiduals_dir=${preprocess_dir}/4_GLMresiduals

if [ ! -e ${extra_regressors_dir}/covar_design.1D ]; then 
echo 'Compiling regressor design file for' ${sub}
if [ -e ${motion_outliers_dir}/*motion_confound ]; then 
paste ${motion_parameters_dir}/*.par ${motion_outliers_dir}/*motion_confound ${brain_covar_dir}/*WMconfound* ${brain_covar_dir}/*CSFconfound* > ${extra_regressors_dir}/covar_design.1D
elif [ ! -e ${motion_outliers_dir}/*motion_confound ]; then 
paste ${motion_parameters_dir}/*.par ${brain_covar_dir}/*WMconfound* ${brain_covar_dir}/*CSFconfound* > ${extra_regressors_dir}/covar_design.1D
fi
fi

# Compute the mean of the time series at each voxel.
if [ ! -e ${GLMresiduals_dir}/${data}_Tmean* ]; then
echo 'Computing voxel-wise temporal mean of functional data for' ${sub}
fslmaths \
${data_loc}/${data} \
-Tmean \
${GLMresiduals_dir}/${data}_Tmean 
fi

if [ ! -e ${GLMresiduals_dir}/*mregress* ]; then
echo 'Regressing out covariates and de-trending for' ${sub}
cd ${GLMresiduals_dir}

# Unzip the .nii.gz output from MCFLIRT above
gzip -f -d ${data_loc}/${data}.nii.gz

3dDeconvolve -input ${data_loc}/${data}.nii \
-ortvec ${extra_regressors_dir}/covar_design.1D lll \
-nobucket \
-polort A \
-errts ${motionfiltered_dir}/${data}_mregress

3dAFNItoNIFTI -prefix ${data}_mregress.nii.gz ${motionfiltered_dir}/${data}_mregress+tlrc

# Clean up
rm ${GLMresiduals_dir}/Decon*
mv ${motionfiltered_dir}/${data}_mregress+* ${GLMresiduals_dir}
cd ${preprocess_dir}
fi

# Add the mean back to the residuals brain map 
if [ ! -e ${GLMresiduals_dir}/*mregress_withmean* ]; then
echo 'Adding the voxel-wise temporal mean back into residuals map for' ${sub}
fslmaths ${GLMresiduals_dir}/${data}_mregress -add ${GLMresiduals_dir}/${data}_Tmean ${GLMresiduals_dir}/${data}_mregress_withmean
fi
data_loc=${GLMresiduals_dir}
data=${data}_mregress_withmean


## Bandpass filter the functional data
if [ ${bandpass_on} = true ]; then 

if [ ! -d ${preprocess_dir}/5_Filter ]; then
mkdir ${preprocess_dir}/5_Filter
fi
Filter_dir=${preprocess_dir}/5_Filter

# Compute the mean of the time series at each voxel.
if [ ! -e ${Filter_dir}/${data}_Tmean* ]; then
echo 'Computing voxel-wise temporal mean of functional data for' ${sub}
fslmaths \
${data_loc}/${data} \
-Tmean \
${Filter_dir}/${data}_Tmean 
fi

if [ ! -e ${Filter_dir}/*bpf.nii* ]; then
gzip -d -f ${data_loc}/${data}.nii.gz
cd ${Filter_dir}
3dBandpass \
-nodetrend \
-band ${fbot} ${ftop} \
-input ${data_loc}/${data}.nii \
-prefix ${data}_bpf
3dAFNItoNIFTI -prefix ${data}_bpf.nii.gz ${data}_bpf+tlrc
cd ${preprocess_dir}
fi 

# Add the mean back to the bandpass-filtered brain map 
if [ ! -e ${Filter_dir}/*bpf_withmean* ]; then
echo 'Adding the voxel-wise temporal mean back into bandpass-filtered map for' ${sub}
fslmaths ${Filter_dir}/${data}_bpf \
-add ${Filter_dir}/${data}_Tmean \
${Filter_dir}/${data}_bpf_withmean
fi
data_loc=${Filter_dir}
data=${data}_bpf_withmean

elif [ ${bandpass_on} = false ]; then 
data_loc=${data_loc}
data=${data}
fi

## Apply a smoothing kernel on the functional data
if [ ${smooth_on} = true ]; then 

if [ ! -d ${preprocess_dir}/6_Smooth ]; then
mkdir ${preprocess_dir}/6_Smooth
fi
Smooth_dir=${preprocess_dir}/6_Smooth

sigma=$(echo "scale=6; ${fwhm} / 2.3548" | bc) # 2.3548 is the FWHM-sigma conversion
if [ ! -e ${Smooth_dir}/*smth* ]; then
echo 'Applying a' ${fwhm}'mm Gaussian kernel on data for' ${sub}
fslmaths  \
${data_loc}/${data} \
-kernel gauss ${sigma} \
-fmean \
${Smooth_dir}/${data}_smth_${fwhm}mm 
data_loc=${Smooth_dir}
data=${data}_smth_${fwhm}mm 
fi

elif [ ${smooth_on} = false ]; then 
data_loc=${data_loc}
data=${data}
fi

## Move filtered data to standard space using 2-step co-registration + normalisation! 
if [ ! -d ${preprocess_dir}/7_Finished ]; then
mkdir ${preprocess_dir}/7_Finished
fi
Finished_dir=${preprocess_dir}/7_Finished

if [ ! -e ${Finished_dir}/${sub}_Prep.nii.gz ]; then
echo 'Moving filtered data to standarad space for' ${sub}
applywarp \
--ref=${standard_location}/MNI152_T1_2mm_brain \
--in=${data_loc}/${data} \
--out=${Finished_dir}/${sub}_Prep.nii.gz \
--premat=${func2highres_dir}/${sub}_flirt_func2highres.mat \
--warp=${highres2std_dir}/${sub}_fnirt_coefficients 
fi

echo 'Finished Preprocessing' ${sub}'!!'
echo 'Finished Preprocessing' ${sub}'!!' > ${preprocess_dir}/finished.txt

done
done
