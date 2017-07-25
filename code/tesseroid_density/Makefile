all:
	python setup.py build_ext --inplace
	mv tesseroid_density/_tesseroid.so .
	rmdir tesseroid_density
	
clean:
	rm -r build
	rm *.pyc
	rm *.so
	rm *.c
