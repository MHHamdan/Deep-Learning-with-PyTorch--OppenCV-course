#!/bin/sh
if [ -d ./build ]; then 
	rm -r ./build 
	mkdir ./build

else 
	mkdir ./build
fi

cd build
cmake ..
cmake --build . --config --Release
cd ..
echo Executing code....!!!!
./build/libtorch-basics
