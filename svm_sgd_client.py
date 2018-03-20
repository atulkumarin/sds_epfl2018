#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:30:41 2018

@author: atul
"""

import grpc
import sgd_svm_pb2_grpc
import sgd_svm_pb2


def scalar_vec_mul(scalar, vec):
    return [scalar*i for i in vec]

def scalar_vec_sum(scalar, vec):
    return [scalar + i for i in vec]

def vec_sum(vec1, vec2):
    return [vec1[i] + vec2[i] for i in range(len(vec1))]

def vec_mul(vec1, vec2):
    num_elements = len(vec1)
    result = 0
    
    for i in range(num_elements):
        result += vec1[i]*vec2[i]
        
    return result
    
    
def mat_mul(mat1, mat2):
    """mat1 is (_n x _m)
       mat2 is (_m x _p )
       resulting matrix will be (_n x _p)
    """
    _n = len(mat1)
    _m = len(mat1[0])
    _p = len(mat2[0])
    
    result = [[0 for x in range(_p)] for y in range(_n)] 
    
    for i in range(_n):
        for j in range(_p):
            for k in range(_m):
                result[i][j] += mat1[i][k]*mat2[k][j]
        
    return result
    

def compute_gradient(param, data_sample, target, lrate=0.2):
    
    if (target*vec_mul(param, data_sample) < 1):
        grad = scalar_vec_mul(-1*target, data_sample)
    else:
        grad = [0 for x in range(len(param))]
    
    grad = vec_sum(grad, scalar_vec_mul(2*lrate, param))

    return grad
    

def parse_response(response):
    target = response.target
    data_sample = response.data_sample
    param = response.param
    
    data_sample = data_sample.split(" ")
    indices = [i.split(":")[0] for i in data_sample]
    data_sample = [i.split(":")[1] for i in data_sample]
    
    #param = param.split(" ")
    param = data_sample
    
    return int(target), indices, [float(a) for a in data_sample], [float(a) for a in param]
    

def next_iterate(stub, grad_update):
        print(type(grad_update))
        response = stub.GetData(grad_update)
        return parse_response(response)
        

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = sgd_svm_pb2_grpc.SGD_SVMStub(channel)
    
    grad_update = sgd_svm_pb2.GradientUpdate(grad_update=' ')
    
    while True:
        print("Yo")
        target, indices, data_sample, param = next_iterate(stub, grad_update)
        grad_update = compute_gradient(param, data_sample, target)
        print("Data Sample = {}, Target = {}, Param = {}, Grad = {}".format(data_sample, 
              target, param, grad_update))
        grad_update = " ".join([(str(i[0]) + ':' + str(i[1])) for i in zip(indices, grad_update)])
        grad_update = sgd_svm_pb2.GradientUpdate(grad_update=grad_update)
 


if __name__ == '__main__':
    run()