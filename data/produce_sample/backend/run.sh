#!/bin/bash
cd /produce-sample/backend
export CUDA_VISIBLE_DEVICES=0
/mntcephfs/data/med/zhanghongbo/anaconda3/envs/chuyi/bin/uvicorn api:app --host 127.0.0.1 --port 8125