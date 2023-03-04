Fortran based 2D FFT benchmark, mimicking what CGYRO does
=========================================================

Usage:
-----

./fortran_test_fft d1 d2 batch

e.g.
./fortran_test_fft 768 190 1152

To compile:
----------

Perlmutter:
ftn -Mpreprocess -craype-verbose -Mdefaultunit -mp -Mstack_arrays  -acc -Minfo=accel -target-accel=nvidia80 -Mcudalib=cufft -r8  -fast -o fortran_test_fft fortran_test_fft.F90

