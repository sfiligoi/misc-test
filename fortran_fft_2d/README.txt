Fortran based 2D FFT benchmark, mimicking what CGYRO does
=========================================================

Usage:
-----

./fortran_test_fft d1 d2 batch

e.g.
./fortran_test_fft 768 190 1152

To compile:
----------

Perlmutter GPU (NVIDIA compiler):
ftn -Mpreprocess -craype-verbose -Mdefaultunit -mp -Mstack_arrays  -acc -Minfo=accel -target-accel=nvidia80 -Mcudalib=cufft -r8  -fast -o fortran_test_fft fortran_test_fft.F90

Perlmutter CPU (CRAY compiler):
ftn -homp -hnoacc  -s real64 -Ofast -I${FFTW_INC}  -o fortran_test_fft fortran_test_fft.F90 -lfftw3_threads -lfftw3f_threads -lfftw3 -lfftw3f -llapack -lblas
# single threaded
ftn -hnoomp -hnoacc  -s real64 -Ofast -I${FFTW_INC}  -o fortran_test_fft fortran_test_fft.F90 -lfftw3 -lfftw3f -llapack -lblas

Frontier GPU (CRAY compiler):
ftn -homp -hacc -DHIPGPU -I${HIPFORT_INC} -hacc_model=auto_async_none:no_fast_addr:no_deep_copy -s real64 -Ofast -o fortran_test_fft fortran_test_fft.F90  -L/opt/rocm-5.1.0/lib -L${HIPFORT_LIB} -lhipfort-amdgcn -lhipfft -lamdhip64


Generic Intel FORTRAN + MKL:
ifort -qopenmp -real-size 64 -Ofast -I${MKLROOT}/include/fftw/ -o fortran_test_fft fortran_test_fft.F90  -qmkl
