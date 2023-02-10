CUDA_VISIBLE_DEVICES=0 /home/lanchuanxin/ProgramFile/cuda-8.0/bin/nvprof --log-file nvprof_time_1.log --system-profiling on --csv -t 1800 -f --continuous-sampling-interval 1 \
python tools/test_lanenet.py --weights_path ./model_culane-71-3/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000 --image_path demo_file/test_img_nvprof.txt --save_dir ./

