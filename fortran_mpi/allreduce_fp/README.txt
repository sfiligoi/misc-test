Simple MPI AllReduce exercise script
====================================

By default it uses OpenMP Target GPU compute and modern mpi_08 library interface.

To use the legacy mpi interface, compile with
 -DUSE_LEGACY_MPI

To use the CPU-only setup, compile with
 -DNO_GPU
