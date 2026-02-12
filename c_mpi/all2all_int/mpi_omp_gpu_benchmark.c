/*
 * Acknowledgement:
 *  Base program genergated with Google Gemini
 *  Some additional manual changes after that
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define WARMUP_ITERS 5

/* Processs wide */
int rank, nranks;


// Helper function to check MPI errors
void check_mpi_error(int ierr, const char *label) {
    if (ierr != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int resultlen;
        MPI_Error_string(ierr, error_string, &resultlen);
        printf("ERROR on Rank %d at %s: %s\n", rank, label, error_string);
        MPI_Abort(MPI_COMM_WORLD, ierr);
    }
}

int main(int argc, char *argv[]) {
    int count, num_iters;
    
    // Timers
    double t_warm_comp = 0.0, t_warm_mpi = 0.0, t_warm_total;
    double t_main_comp = 0.0, t_main_mpi = 0.0, t_main_total;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // 1. CLI Arguments
    if (argc < 3) {
        if (rank == (nranks - 1)) {
            printf("Usage: mpirun ./mpi_omp_gpu_benchmark <count_per_rank> <main_iterations>\n");
        }
        MPI_Finalize();
        return 1;
    }

    count = atoi(argv[1]);
    num_iters = atoi(argv[2]);
    long total_elements = (long)count * nranks;

    int *device_sendbuf = (int *)malloc(total_elements * sizeof(int));
    int *device_recvbuf = (int *)malloc(total_elements * sizeof(int));

    // Initial GPU Map
#ifndef NO_GPU
    #pragma omp target enter data map(alloc: device_sendbuf[0:total_elements], device_recvbuf[0:total_elements])
    #pragma omp target teams distribute parallel for
#else
    #pragma omp parallel for
#endif
    for (int i = 0; i < total_elements; i++) {
        device_sendbuf[i] = rank;
        device_recvbuf[i] = 0;
    }

    // --- 2. TIMED WARMUP PHASE ---
    if (rank == (nranks - 1)) printf(">>> Starting Warmup (%d iters)\n", WARMUP_ITERS);
    
    check_mpi_error(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");
    t_warm_total = MPI_Wtime();

    for (int iter = 0; iter < WARMUP_ITERS; iter++) {
        double t0 = MPI_Wtime();
#ifndef NO_GPU
        #pragma omp target teams distribute parallel for
#else
        #pragma omp parallel for
#endif
        for (int i = 0; i < total_elements; i++) {
            device_sendbuf[i] = device_sendbuf[i] + 1;
        }
        double t1 = MPI_Wtime();
        t_warm_comp += (t1 - t0);

	int ierr = -1;
        t0 = t1;
#ifndef NO_GPU
        #pragma omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
        {
            ierr = MPI_Alltoall(device_sendbuf, count, MPI_INT, 
                                device_recvbuf, count, MPI_INT, MPI_COMM_WORLD);
        }
        t1 = MPI_Wtime();
        t_warm_mpi += (t1 - t0);
        check_mpi_error(ierr, "Warmup Alltoall");
    }
    t_warm_total = MPI_Wtime() - t_warm_total;

    // Basic check that recvbuffer is being updated in GPU memory
    {
      int recvmax = -1;
#ifndef NO_GPU
      #pragma omp target teams distribute parallel for reduction(max:recvmax) map(tofrom:recvmax)
#endif
      for (int i = 0; i < total_elements; i++) {
        if (recvmax < device_recvbuf[i]) recvmax = device_recvbuf[i];
      }

      if (recvmax < 1) {
        printf("Recv error in warmup, >0 expected, found %d rank= %d\n", recvmax, rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }

    // --- 3. TIMED MAIN PHASE ---
    if (rank == (nranks - 1)) printf(">>> Starting Main Loop (%d iters)\n", num_iters);
    
    check_mpi_error(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");
    t_main_total = MPI_Wtime();

    for (int iter = 0; iter < num_iters; iter++) {
        double t0 = MPI_Wtime();
#ifndef NO_GPU
        #pragma omp target teams distribute parallel for
#else
        #pragma omp parallel for
#endif
        for (int i = 0; i < total_elements; i++) {
            // C logic uses 0-indexing, handling the 'i=0' (Fortran i=1) case
            if (i > 0) {
                device_sendbuf[i] = (device_sendbuf[i] + (device_recvbuf[i-1] + 2 * device_recvbuf[i]) / 3) % 1234567;
            } else {
                device_sendbuf[i] = (device_sendbuf[i] + (device_recvbuf[total_elements-1] + 2 * device_recvbuf[i]) / 3) % 1234567;
            }
        }
        double t1 = MPI_Wtime();
        t_main_comp += (t1 - t0);

	int ierr = -1;
        t0 = t1;
#ifndef NO_GPU
        #pragma omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
        {
            ierr = MPI_Alltoall(device_sendbuf, count, MPI_INT, 
                                device_recvbuf, count, MPI_INT, MPI_COMM_WORLD);
        }
        t1 = MPI_Wtime();
        t_main_mpi += (t1 - t0);
        check_mpi_error(ierr, "Main Alltoall");
    }
    t_main_total = MPI_Wtime() - t_main_total;

    // --- 4. REPORTING ---
    if (rank == (nranks - 1)) {
        printf("\n---------------- PERFORMANCE SUMMARY ----------------\n");
        printf(" Ranks: %d | Count: %d\n", nranks, count);
        printf(" Warmup Total: %12.6f s\n", t_warm_total);
        printf(" Main Total:   %12.6f s\n", t_main_total);
        printf("\n AVERAGE TIME PER ITERATION (Steady State vs Warmup)\n");
        printf(" Compute: %12.6f s  (Warmup: %12.6f s)\n", t_main_comp/num_iters, t_warm_comp/WARMUP_ITERS);
        printf(" MPI:     %12.6f s  (Warmup: %12.6f s)\n", t_main_mpi/num_iters, t_warm_mpi/WARMUP_ITERS);
        
        double total_bytes = (double)total_elements * sizeof(int) * nranks;
        double avg_mpi_time = t_main_mpi / num_iters;
        double gb = 1024.0 * 1024.0 * 1024.0;

        printf("\n Steady State MPI Total Bandwidth: %10.2f GB/s\n", total_bytes / (avg_mpi_time * gb));
        printf(" Steady State MPI Per process Bandwidth: %10.2f GB/s\n", (total_bytes / nranks) / (avg_mpi_time * gb));
        printf("-----------------------------------------------------\n");
    }

    // Final check to ensure MPI worked
#ifndef NO_GPU
    #pragma omp target update from(device_sendbuf[0:3], device_recvbuf[0:3])
#endif

    if (rank == (nranks - 1)) {
        printf("Sample Send Buffer elements (Index 0-2): %d %d %d\n", device_sendbuf[0], device_sendbuf[1], device_sendbuf[2]);
        printf("Sample Recv Buffer elements (Index 0-2): %d %d %d\n", device_recvbuf[0], device_recvbuf[1], device_recvbuf[2]);
    }

#ifndef NO_GPU
    #pragma omp target exit data map(release: device_sendbuf[0:total_elements], device_recvbuf[0:total_elements])
#endif

    free(device_sendbuf);
    free(device_recvbuf);
    MPI_Finalize();

    return 0;
}
