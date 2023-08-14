ACC vs OMP comparison, mimicking partially what CGYRO does
==========================================================

Usage:
-----

./test_ompgpu_acc
./test_ompgpu_omp

To compile:
----------

Perlmutter GPU (NVIDIA compiler):
ftn -module . -Mpreprocess -DUSE_INLINE -craype-verbose -Mdefaultunit -mp -Mstack_arrays  -acc -Minfo=accel -target-accel=nvidia80 -fast -o test_ompgpu_acc test_ompgpu.F90
ftn -module . -Mpreprocess -DUSE_INLINE -craype-verbose -Mdefaultunit -Mstack_arrays  -mp=gpu -DOMPGPU -Minfo=mp,accel -target-accel=nvidia80 -fast -o test_ompgpu_omp test_ompgpu.F90
#ftn -module . -Mpreprocess -DUSE_INLINE -craype-verbose -Mdefaultunit -Mstack_arrays -fast -o test_ompgpu_cpu test_ompgpu.F90


Frontier GPU (CRAY compiler):
ftn -J . -homp -hacc -s real64 -Ofast -o test_ompgpu_acc test_ompgpu.F90
ftn -J . -homp -DOMPGPU -s real64 -Ofast -o test_ompgpu_omp test_ompgpu.F90

