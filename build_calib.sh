#!/bin/bash
mkdir -p build
cd build
mkdir -p calib
cd calib && cmake ../../calib
make
