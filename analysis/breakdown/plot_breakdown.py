import csv
import pandas as pd
egien_type={
    'relu':'relu',
    'Add': 'Add',
    'winograd': 'convolve',
    'convolve':  'convolve',
    'gemv2N':'convolve',
    'pool':'pool',
    'gemm':'convolve',
    'norm':'norm',
    'CatArrayBatchedCopy':'memory',
    'scalar_product_op':'multiply',
    'scalar_sum_op':'Add'


}




egien_type2={
        'scalar_product_op':'multiply',
        'scalar_sqrt_op':'sqrt',
        'scalar_difference_op':'compare',
        'scalar_sum_op':'sum',
       # 'scalar_constant_op':'data',
        'scalar_quotient_op':'quotient',
     #   'TensorStridingSlicingOp':'data',
      #  'TensorSlicingOp':'data',
    #    'TensorShufflingOp':'data',
    #    'TensorReverseOp':'data',
        'scalar_sigmoid_op':'sigmoid',
        'scalar_sigmoid_gradient_op':'sigmoid',
        'scalar_opposite_op':'opposite',
        'scalar_tanh_op':'tanh',
        'scalar_tanh_gradient_op':'tanh',
        'scalar_boolean_and_op':'logical',
        'ArgMaxTupleReducer':'argmax',
    #    'TensorGeneratorOp':'data',
    #    'TensorSelectOp':'data',
     #   'TensorConversionOp':'data',
     #   'TensorBroadcastingOp':'data',
        'less':'compare',
        'scalar_square_op':'square',
        'scalar_exp_op':'exp',
     #   'scalar_inverse_op':'data',
        'scalar_log1p_op':'log',
        'equal_to':'compare',
        'greater':'compare',
    #    'scalar_const_op':'data',
        'scalar_boolean_not_op':'logical',
        'scalar_max_op':'max',
   #     'scalar_floor_op':'data',
        'scalar_min_op':'min',
        'scalar_binary_pow_op_google':'logical',
    #    'TensorPaddingOp':'data',
        'scalar_rsqrt_op':'sqrt'
        #'Eigen::TensorMap<Eigen::Tensor<':'initialize data'
 }
def rec_egien(filename):
    print(filename)
    res = {}
    for i in egien_type.values():
        res[i] = 0
    res['data']=0
    res['others']=0
    with open(filename,"r") as f:
        l = f.readlines()
        for i in l:
            if 1==1:
            #if("GPU activities" in i):
                if 1==1:
                #if ("void Eigen" in i):
                    count = 0
                    for j in egien_type.keys():
                        if((j in i)and (count==0)):
                            #print(i)
                            #print(egien_type[j])
                            count = count + 1
                            v = float(i.split(",")[1])
                            res[egien_type[j]] = res[egien_type[j]] + v
                    if(count == 0):
                        v = float(i.split(",")[1])
                        res['data'] = res['data'] + v
                    if(count >1):
                        print(i)
                else:
                    v = float(i.split(",")[1])
                    res['others'] = res['others'] +v
    return res

result1 = rec_egien("tran_nvprof/nvprof_time_1.log")
result2 = rec_egien("tran_nvprof/nvprof_time_2.log")
result3 = rec_egien("tran_nvprof/nvprof_time_3.log")
result4 = rec_egien("tran_nvprof/nvprof_time_4.log")

with open('dict.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['type','Lane','detection','light','sign'])#,'text','text classification','ocr'])
    for key in result1.keys():
       writer.writerow([key, result1[key], result2[key], result3[key], result4[key]]) #,result5[key],result6[key],result7[key]])
    