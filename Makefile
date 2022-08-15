build:
	mkdir build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
	cd build
	make -C build
	./build/raytracer
clean:
	rm -rf build
	rm *ppm
	rm -rf raytracer.dSYM
debug:
	c++ src/main.cc -g -O2 -o raytracer -std=c++17
	./raytracer
	
