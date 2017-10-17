#!/usr/bin/env bash
caffe train --solver=./solver.prototxt --weights=train_iter_40000.caffemodel --gpu=$1
