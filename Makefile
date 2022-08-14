build:
	mkdir build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
	cd build
	make -C build
	./build/raytracer
clean:
	rm -rf build
debug:
	c++ src/main.cc -g -o raytracer -std=c++17
	./raytracer
	
