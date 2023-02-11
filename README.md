# Edge AIBench V2.0
**The Edge AIBench V2.0 is a benchmarking tool for Edge AIBench systems**

## Workload

The Edge AIBench V2.0 includes 4 workloads

No1: The *object detection* task uses YOLOv5 as the deep learning network model and selected BDD100K as the
dataset, which contains 100,000 labeled HD datasets.

No2: The *traffic light classification* task uses a CNN model as the deep learning network model. It uses the Nexar dataset as the real-world dataset, which contains 18,659 abeled training datasets containing traffic signal images and 500,000 test data images.

No3: The *road sign classification* task uses a CNN deep learning model based on the LeNet framework and the German Traffic Sign Recognition Benchmark (GTRB) as the dataset, and the task classifies 43 classes of traffic signs.

No4: 

## Measurement

*Tail Latency of the Whole Scenario*

Autonomous driving scenarios are very demanding in terms of latency, so we choose latency as a quality of service metric for this scenario benchmark. We break down the whole scenario latency to each task module to discover which modules are the primary contributors to latency in the whole scenario


*Hotspot Function Analysis*

Because the majority of the tasks in autonomous driving scenarios employ deep learning models, which require the high performance of the onboard chips, as a result, we decomposed the execution time of GPUs using the profiling tool [nvprof]( https://docs.nvidia.com/cuda/profiler-usersguid) offered by Nvidia. Then we analyze those hotspot functions.

## Reference:

https://github.com/Offpics/traffic-light-classifier-tensorflow

https://github.com/vijaykrishnay/Udacity_TrafficSignClassifier

https://github.com/cardwing/Codes-for-Lane-Detection

https://github.com/williamhyin/yolov5s_bdd100k
