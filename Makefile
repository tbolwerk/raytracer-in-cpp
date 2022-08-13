build:
	mkdir build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
	cd build
	make -C build
	./build/raytracer
clean:
	rm -rf build
	rm -rf debug
debug:
	mkdir debug
	c++ src/main.cc -o build/raytracer -std=c++17
	./build/raytracer
