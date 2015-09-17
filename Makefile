all:
	mkdir -p bin
	gcc src/*.c -o bin/dot-product -std=gnu99 -g -Wall -pedantic -lrt -lm -Ofast -fno-unsafe-math-optimizations -march=core-avx-i
	icc src/*.c -o bin/dot-product-icc -std=gnu99 -g -lrt -lm -O3 -fp-model precise -xAVX
	gcc src/*.c -o bin/dot-product-unsafe -std=gnu99 -g -Wall -pedantic -lrt -lm -Ofast -march=core-avx-i
	icc src/*.c -o bin/dot-product-unsafe-icc -std=gnu99 -g -lrt -lm -O3 -xAVX

parallel:
	mkdir -p bin
	gcc src/*.c -o bin/dot-product -std=gnu99 -g -Wall -pedantic -lrt -lm -Ofast -fno-unsafe-math-optimizations -march=core-avx-i -fopenmp

run:
	./bin/dot-product | tee res/gcc
	./bin/dot-product-icc | tee res/icc

clean:
	rm -r bin
