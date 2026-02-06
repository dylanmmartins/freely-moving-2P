# Freely moving two-photon calcium imaging


## Setup
To install this as a package within a conda environment, run `pip install -e .` in the directory of this repository.

To create the conda environment from the explicit package versions in "spec-file.txt", run `conda create --name fm1 --file spec-file.txt`. New specification files can be created with `conda list --explicit > spec-file.txt`.

## Use guide

### Resulting data

The data written to disk from a recording, before any analysis is run, are:
* a tif stack of two-photon data written by Scanimage (e.g., 'file_00001.tif'). This will have been acquired at 7.50 Hz and should have the dimensions (512, 512, n_frames)
* a mp4 video of the topdown camera, also acquired at 7.50 Hz and of the same length (in frame count) as the tif stack (e.g., fm1_0001.mp4)
* a json written alongside the topdown video (Experiment.json) which stores the settings use for gain, exposure, etc. This is not used in the analysis.
* The raw eyecam video, which must be longer in physical time than the topdown video and tif stack. It will have been acquired at 30 Hz but as an interlaced video that can be deinterlaced to 60 Hz. It should be



### Analyze 2P data

_Denoise the image stack_:

This is more important for axonal imaging than for somatic imaging, because axonal imaging uses higher laser power and has a worse SNR. There are vertical and horizonal bands of resonance scanner noise that sweep across the image (i.e,. they are not static in position). They sweep slowly both leftward and rightward, with changing overlap over time. On one of our two scopes, they also exist as horizontal bands that sweep both upward and downward. To subtract this, run `python -m fm2p.denoise_tif` and select the raw tif stack written by Scanimage in the dialog box that opens. This code is memory intensive and needs to be run on a computer with ~128 GB RAM (but it runs pretty quickly, even on long recordings). A PDF of diagnostic figures and two tif files will be written to disk: one is the computed noise pattern and the other is a denoised tif stack. Both will have the same shape as the origional tif image stack. A diagnostic video can be made by running `python -m fm2p.denoise_tif -makevid True`.

_Segment somatic imaging:_

Analyze the two-photon imaging .tif stack using Suite2p by running `python -m suite2p` in a conda environment with suite2p version 0.14.2 installed (suite2p is not in the fm1 conda environment). For the GCaMP6s sensor, use `tau=1.2`. Use the appropriate sample rate, `fs=7.50`. Load the model, "~/model_data/mini2P_soma_classifier.npy", and apply it by adjusting the threshold to any new value and then back to `0.5`. Refine further as needed.

If there is a sparse noise recording for this experiment (or just multiple recordings from the same session for which you would like to tgrack cells across seperate tif stacks), run suite2p without moving each tif stack to individual recording directories. Once suite2p is done running on the combined recordings, split them back out by running `python -m fm2p.split_suite2p`. Dialog boxes will open in which you (1) select the suite2p directory that contains the .npy files for the combined sparse noise + freely moving recordings; (2) The first tif stack in the merged data, which should be the freely moving tif as long as freely moving was recorded before sparse noise and scanimage gave a lower number to the freely moving tif stack; (3) The save directory for the first recording; (4) The save directory for the second recording. This only works for two recordings, and there is not currently a way to split more than two (TODO: should I write this?).

_Segment axonal imaging_:

Use the Goard lab two-photon calcium post processing pipeline repository [here](https://github.com/ucsb-goard-lab/Two-photon-calcium-post-processing), which I run with Matlab 2023b. Run the function `A_ProcessTimeSeries.m` without image registration. Then, run `B_DefineROI.m`, perform "Activity Map" segmentation with default values except for the "minimum pixel size" which should be changed to 5. Next, run `C_ExtractDFF.m` choosing "Local Neuropil Subtraction" and choosing "Yes" to "Weight subtraction to minimize signal-noise correlation?"

### Create the config file

Make a copy of the file "~/config.yaml", save it in the main directory of the recording, i.e., ".../250101_DMM_DMM000_pillar/config.yaml". Add the recording directory to the file (in the `spath` field). Other fields can usually stay as-is.

### Preprocesss the recording

Activate the conda environment (`conda activate fm1`). Preprocess the recording by running `python -m fm2p.preprocess -cfg  K:/Mini2P/250101_DMM_DMM000_pillar/config.yaml` with the path to the config file create in the previous step. If you don't include the `-cfg` flag, a dialog box will open in which you can select the config file.

During this process, three dialog boxes will open showing an example frame from the top-down video.

In the first window, four blue points are placed on the corners of the arena. Once a point is placed, you cannot adjust it. They *must* be placed in the following order: top-left, top-right, bottom-left, bottom-right.

In the next window, eight red points are placed around the perimeter of the checkerboard pillar in the arena. They should be placed in continuous order, such that if the first point is at the top of the arena, the next should be in either the top-left or top-right corner and continue around the perimeter of the pillar in counter-clockwise or clockwise order, respectively. Place these points on the edge of the pillar at its highest point, ignoring that the base of the pillar may be in a different position depending on how close to the edge of the field of view the pillar was in this recording.

The third window will show an orange line tracing the points placed in the previous window. Click on the orange line and drag it so that the circle is now over the base of the pillar instead of at the top of the pillar. For a pillar in the center of the arena, these will be identical positions. For a pillar near the edge of the arena, there may be a large distance between these two positions. When you're happy with its position, close this window.

This pipeline will:
* deinterlace the eyecamera video
* run DeepLabCut to measure the edges of the animals pupil
* run DeepLabCut to track the animal's position from the top-down camera
* measure the animal's position and orientation in the arena, align the behavioral measures from the eyecamera video to the 2P and top-down data using the TTL voltages read in through Bonsai
* fit an ellipse to the pupil and measure the pupil orientation via ellipse tilt
* read in Suite2p outputs, calculate dF/F, run the OASIS algorithm to infer spikes from fluorescence data, and create timestamps for each two-photon frame
* interpolate pupil orientations to two-photon and top-down timestamps
* measure the position of the arena's pillar in retinocentric and egocentric coordinates

For sparse noise recordings, it will do the subset of these steps that makes sense.

A single .h5 file will be written in the directory of the recording with all preprocessed data.

### Sparse noise visual spike-triggered averages

To map a spike-triggered average receptive field for each cell, run `python -m fm2p.sparse_noise_mapping -method splits -path /home/..../...preproc.h5`. Once the STAs are calculated, they are manually sorted into reliable/unreliable responses. Run `python -m fm2p.revieW_STAs` and select the "sparse_noise.h5" file for the desired recording. A window will open that has three panels. Left to right, they are: full STA, first split (i.e., half the data used to calculate an STA), second split. If it looks like a nice receptive field in the first panel and the two splits are in agreement with one another, press the right arrow key. Otherwise, press the left arrow key. Continue this until all cells are reviewed. Once completed, a gaussian will be fit to all of the real receptive fields and the properties will be saved out into the resulting ".npz" file.

### Simple tuning curves for eye/head movement variables

This code works for recordings with an IMU as well as those without an IMU.