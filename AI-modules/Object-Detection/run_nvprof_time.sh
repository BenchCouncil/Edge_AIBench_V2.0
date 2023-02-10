CUDA_VISIBLE_DEVICES=0 /home/lanchuanxin/ProgramFile/cuda-8.0/bin/nvprof --log-file nvprof_time_2.log --system-profiling on --csv -t 1800 -f --continuous-sampling-interval 1 \
python detect.py --weights runs/exp1_yolov5s_bdd/weights/best_yolov5s_bdd.pt  --source bdd100k/images/test_nvprof  --save-txt
