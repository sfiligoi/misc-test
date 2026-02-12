/*
 * Acknowledgement:
 *  Base program genergated with Google Gemini
 *  Some additional manual changes after that
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

void check_mpi_error(int ierr, const char *label, int rank) {
    if (ierr != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int resultlen;
        MPI_Error_string(ierr, error_string, &resultlen);
        fprintf(stderr, "ERROR on Rank %d at %s: %s\n", rank, label, error_string);
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }
}

int main(int argc, char *argv[]) {
    int rank, nranks, ierr;
    int count, num_iters;
    const int warmup_iters = 5;
    double recvmax;

    // Timers
    double t0, t1;
    double t_warm_comp = 0.0, t_warm_mpi = 0.0, t_warm_total = 0.0;
    double t_main_comp = 0.0, t_main_mpi = 0.0, t_main_total = 0.0;

    double *device_sendbuf, *device_recvbuf;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // 1. CLI Arguments
    if (argc < 3) {
        if (rank == 0) printf("Usage: mpirun ./mpi_omp_gpu_allreduce <count> <iterations>\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    count = atoi(argv[1]);
    num_iters = atoi(argv[2]);

    device_sendbuf = (double*)malloc(count * sizeof(double));
    device_recvbuf = (double*)malloc(count * sizeof(double));

    // Initial GPU Map and Data Setup
#ifndef NO_GPU
    #pragma omp target enter data map(alloc: device_sendbuf[0:count], device_recvbuf[0:count])
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for (int i = 0; i < count; i++) {
        device_sendbuf[i] = 1.0 + 0.0001 * (i + 1);
        device_recvbuf[i] = 0.0;
    }

    // --- 2. TIMED WARMUP PHASE ---
    if (rank == 0) printf(">>> Starting Warmup (%d iters)\n", warmup_iters);
    MPI_Barrier(MPI_COMM_WORLD);
    t_warm_total = MPI_Wtime();

    for (int iter = 0; iter < warmup_iters; iter++) {
        t0 = MPI_Wtime();
#ifndef NO_GPU
        #pragma omp target teams distribute parallel for
#else
        #pragma omp parallel for
#endif
        for (int i = 0; i < count; i++) {
            device_sendbuf[i] = device_sendbuf[i] * 1.000001 + device_recvbuf[i] / count;
        }
        t1 = MPI_Wtime(); 
        t_warm_comp += (t1 - t0);

        t0 = t1;
#ifndef NO_GPU
        #pragma omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
        {
            ierr = MPI_Allreduce(device_sendbuf, device_recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        t1 = MPI_Wtime();
        t_warm_mpi += (t1 - t0);
        check_mpi_error(ierr, "Warmup AllReduce", rank);
    }
    t_warm_total = MPI_Wtime() - t_warm_total;

    // Basic check that recvbuffer is being updated in GPU memory
    recvmax = -1.0;
#ifndef NO_GPU
    #pragma omp target teams distribute parallel for reduction(max:recvmax) map(tofrom:recvmax)
#endif
    for (int i = 0; i < count; i++) {
        if (recvmax < device_recvbuf[i]) recvmax = device_recvbuf[i];
    }

    if (recvmax < 1.0) {
        printf("Recv error in warmup, >0 expected, found %f rank= %d\n", recvmax, rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- 3. TIMED MAIN PHASE ---
    if (rank == 0) printf(">>> Starting Main Loop (%d iters)\n", num_iters);
    MPI_Barrier(MPI_COMM_WORLD);
    t_main_total = MPI_Wtime();

    for (int iter = 0; iter < num_iters; iter++) {
        t0 = MPI_Wtime();
#ifndef NO_GPU
        #pragma omp target teams distribute parallel for
#else
        #pragma omp parallel for
#endif
        for (int i = 0; i < count; i++) {
            device_sendbuf[i] = device_sendbuf[i] * 1.000001 + device_recvbuf[i] / count;
        }
        t1 = MPI_Wtime();
        t_main_comp += (t1 - t0);

        t0 = t1;
#ifndef NO_GPU
        #pragma omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
        {
            ierr = MPI_Allreduce(device_sendbuf, device_recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        t1 = MPI_Wtime();
        t_main_mpi += (t1 - t0);
        check_mpi_error(ierr, "Main AllReduce", rank);
    }
    t_main_total = MPI_Wtime() - t_main_total;

    // --- 4. REPORTING ---
    if (rank == (nranks - 1)) {
        printf("\n---------------- ALLREDUCE PERFORMANCE ----------------\n");
        printf(" Ranks: %d | Count: %d\n", nranks, count);
        printf(" Main Total:    %f s\n", t_main_total);
        printf("\n AVERAGE TIME PER ITERATION\n");
        printf(" Compute: %f s  (Warmup: %f s)\n", t_main_comp / num_iters, t_warm_comp / warmup_iters);
        printf(" MPI:     %f s  (Warmup: %f s)\n", t_main_mpi / num_iters, t_warm_mpi / warmup_iters);

        double throughput = ((double)count * 8.0) / ((t_main_mpi / num_iters) * 1024.0 * 1024.0 * 1024.0);
        printf("\n Avg MPI Throughput: %10.2f GB/s\n", throughput);
        printf("-------------------------------------------------------\n");
    }

    // Final check to ensure MPI worked
#ifndef NO_GPU
    #pragma omp target update from(device_sendbuf[0:3], device_recvbuf[0:3])
#endif
    if (rank == (nranks - 1)) {
        printf("Sample Send Buffer elements (Index 0-2): %f %f %f\n", device_sendbuf[0], device_sendbuf[1], device_sendbuf[2]);
        printf("Sample Recv Buffer elements (Index 0-2): %f %f %f\n", device_recvbuf[0], device_recvbuf[1], device_recvbuf[2]);
    }

#ifndef NO_GPU
    #pragma omp target exit data map(release: device_sendbuf[0:count], device_recvbuf[0:count])
#endif
    free(device_sendbuf);
    free(device_recvbuf);

    MPI_Finalize();
    return 0;
}
