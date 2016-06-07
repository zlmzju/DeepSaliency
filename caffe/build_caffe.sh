#!/bin/sh 
cp Makefile.config.centos Makefile.config
make -j8;make pycaffe
