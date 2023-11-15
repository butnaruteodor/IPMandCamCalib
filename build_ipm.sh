#!/bin/bash
mkdir -p build
cd build
mkdir -p ipm
cd ipm && cmake ../../ipm
make
