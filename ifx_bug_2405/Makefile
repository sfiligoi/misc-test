all: cgyro_test_bug cgyro_test_ok


cgyro_globals.o: cgyro_globals.F90
	ifx -gen-interfaces -fiopenmp -fopenmp-targets=spir64 -real-size 64 -qmkl=parallel -fpp -free -g -O2 -c cgyro_globals.F90

cgyro_freq_bug.o: cgyro_freq.F90 cgyro_globals.o
	# using -O2 creates bad code
	ifx -gen-interfaces -fiopenmp -fopenmp-targets=spir64 -real-size 64 -qmkl=parallel -fpp -free -g -O2 -c cgyro_freq.F90 -o cgyro_freq_bug.o

cgyro_freq_ok.o: cgyro_freq.F90 cgyro_globals.o
	# using -O0 works
	ifx -gen-interfaces -fiopenmp -fopenmp-targets=spir64 -real-size 64 -qmkl=parallel -fpp -free -g -O0 -c cgyro_freq.F90 -o cgyro_freq_ok.o

cgyro_test.o: cgyro_test.f90 cgyro_globals.o
	ifx -gen-interfaces -fiopenmp -fopenmp-targets=spir64 -real-size 64 -qmkl=parallel -fpp -free -g -O2 -c cgyro_test.f90

cgyro_test_bug: cgyro_test.o cgyro_freq_bug.o cgyro_globals.o
	# using the fully optimized code
	ifx -o cgyro_test_bug cgyro_test.o cgyro_freq_bug.o cgyro_globals.o

cgyro_test_ok: cgyro_test.o cgyro_freq_ok.o cgyro_globals.o
	# using the fully optimized code
	ifx -o cgyro_test_ok cgyro_test.o cgyro_freq_ok.o cgyro_globals.o

clean:
	rm -f *.o *mod* cgyro_test_bug cgyro_test_ok
