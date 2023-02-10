CUDA_VISIBLE_DEVICES=0 /home/lanchuanxin/ProgramFile/cuda-8.0/bin/nvprof --log-file nvprof_time_4.log --system-profiling on --csv -t 1800 -f --continuous-sampling-interval 1 \
python test_cxlan.py
