CC=c++

build:
	mkdir build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
	cd build
	make -C build
	./build/raytracer
clean:
	rm -rf build
	rm -rf debug
	rm *ppm
	rm -rf raytracer.dSYM
debug:
	mkdir debug
	$(CC) src/main.cc -g -O2 -o debug/raytracer -std=c++17 -Wall -Wextra -pedantic
	./debug/raytracer

	
