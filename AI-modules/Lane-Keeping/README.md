Tensorflow version of SCNN in CULane.

## Installation
    conda create -n tensorflow_gpu pip python=3.5
    source activate tensorflow_gpu
    pip install --upgrade tensorflow-gpu==1.3.0
    pip3 install -r lane-detection-model/requirements.txt 


## Pre-trained model for testing
Download the pre-trained model [here](https://drive.google.com/open?id=1-E0Bws7-v35vOVfqEXDTJdfovUTQ2sf5).

## Test
   ./run.sh

Note that path/to/image_name_list should be like [test_img.txt](./lane-detection-model/demo_file/test_img.txt). Now, you get the probability maps from our model. To get the final performance, you need to follow [SCNN](https://github.com/XingangPan/SCNN) to get curve lines from probability maps as well as calculate precision, recall and F1-measure.

## NVProf
   ./run_nvprof_time.sh
   ./run_nvprof_metric.sh
