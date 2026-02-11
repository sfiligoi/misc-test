!
! Acknowledgement:
!  Base program genergated with Google Gemini
!  Some additional manual changes after that
!
program mpi_omp_gpu_benchmark
#ifndef USE_LEGACY_MPI
    use mpi_f08
    use omp_lib
#else
    use mpi
#endif
    implicit none

    integer :: rank, nranks, i, iter, ierr, arg_count
    integer :: count, num_iters
    integer, parameter :: warmup_iters = 5
    integer :: total_elements
    character(len=32) :: arg_str
    
    ! Timers
    real(8) :: t0, t1
    real(8) :: t_warm_comp, t_warm_mpi, t_warm_total
    real(8) :: t_main_comp, t_main_mpi, t_main_total
    
    integer, dimension(:), allocatable :: device_sendbuf, device_recvbuf

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

    ! 1. CLI Arguments
    arg_count = command_argument_count()
    if (arg_count < 2) then
        if (rank == 0) print *, "Usage: mpirun ./mpi_omp_gpu_benchmark <count_per_rank> <main_iterations>"
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    call get_command_argument(1, arg_str); read(arg_str, *) count
    call get_command_argument(2, arg_str); read(arg_str, *) num_iters

    total_elements = count * nranks
    allocate(device_sendbuf(total_elements), device_recvbuf(total_elements))

    ! Initial GPU Map
#ifndef NO_GPU
    !$omp target enter data map(alloc: device_sendbuf, device_recvbuf)
    !$omp target teams distribute parallel do
#else
    !$omp parallel do
#endif
    do i = 1, total_elements
        device_sendbuf(i) = rank
    end do

    ! Initialize timers
    t_warm_comp = 0.0; t_warm_mpi = 0.0; t_main_comp = 0.0; t_main_mpi = 0.0

    ! --- 2. TIMED WARMUP PHASE ---
    ! Similar to main below, but just to cleanly separate startup and steady state performance
    if (rank == 0) print '(A, I0, A)', ">>> Starting Warmup (", warmup_iters, " iters)"
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    call check_mpi_error(ierr, "Warmup Barrier")
    t_warm_total = MPI_Wtime()

    do iter = 1, warmup_iters
        t0 = MPI_Wtime()
#ifndef NO_GPU
        !$omp target teams distribute parallel do
#else
    !$omp parallel do
#endif
        do i = 1, total_elements
            device_sendbuf(i) = device_sendbuf(i)+1
        end do
        t1 = MPI_Wtime(); t_warm_comp = t_warm_comp + (t1 - t0)

        t0 = t1
#ifndef NO_GPU
        !$omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
            call MPI_Alltoall(device_sendbuf, count, MPI_INTEGER, &
                              device_recvbuf, count, MPI_INTEGER, MPI_COMM_WORLD, ierr)
#ifndef NO_GPU
        !$omp end target data
#endif
        t1 = MPI_Wtime(); t_warm_mpi = t_warm_mpi + (t1 - t0)
        call check_mpi_error(ierr, "Warmup Alltoall")
    end do
    t_warm_total = MPI_Wtime() - t_warm_total

    ! --- 3. TIMED MAIN PHASE ---
    if (rank == 0) print '(A, I0, A)', ">>> Starting Main Loop (", num_iters, " iters)"
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    call check_mpi_error(ierr, "Main Barrier")
    t_main_total = MPI_Wtime()

    do iter = 1, num_iters
        ! some dummy compute, to mimick real-life workflows
        t0 = MPI_Wtime()
#ifndef NO_GPU
        !$omp target teams distribute parallel do
#else
    !$omp parallel do
#endif
        do i = 1, total_elements
          if (i>1) then
            device_sendbuf(i) = (device_recvbuf(i-1)+2*device_recvbuf(i))/3;
          else
            device_sendbuf(i) = (device_recvbuf(total_elements)+2*device_recvbuf(i))/3;
          endif
        end do
        t1 = MPI_Wtime(); t_main_comp = t_main_comp + (t1 - t0)

        t0 = t1;
#ifndef NO_GPU
        !$omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
            call MPI_Alltoall(device_sendbuf, count, MPI_INTEGER, &
                              device_recvbuf, count, MPI_INTEGER, MPI_COMM_WORLD, ierr)
#ifndef NO_GPU
        !$omp end target data
#endif
        t1 = MPI_Wtime(); t_main_mpi = t_main_mpi + (t1 - t0)
        call check_mpi_error(ierr, "Main Alltoall")
    end do
    t_main_total = MPI_Wtime() - t_main_total

    ! --- 4. REPORTING ---
    if (rank == 0) then
        print '(/, A)', "---------------- PERFORMANCE SUMMARY ----------------"
        print '(A, I0, A, I0)', " Ranks: ", nranks, " | Count: ", count
        print '(A, F12.6, A)',  " Warmup Total: ", t_warm_total, " s"
        print '(A, F12.6, A)',  " Main Total:   ", t_main_total, " s"
        print '(/, A)', " AVERAGE TIME PER ITERATION (Steady State vs Warmup)"
        print '(A, F12.6, A, F12.6, A)', " Compute: ", t_main_comp/num_iters, " s  (Warmup: ", t_warm_comp/warmup_iters, " s)"
        print '(A, F12.6, A, F12.6, A)', " MPI:     ", t_main_mpi/num_iters,  " s  (Warmup: ", t_warm_mpi/warmup_iters,  " s)"
        
        ! Bandwidth based on Main Phase
        print '(/, A, F10.2, A)', " Steady State MPI Total Bandwidth: ", &
            (real(total_elements,8) * 4.0d0 * nranks) / (t_main_mpi/num_iters * 1024.0d0**3), " GB/s"
        print '(A, F10.2, A)', " Steady State MPI Per process Bandwidth: ", &
            (real(total_elements,8) * 4.0d0) / (t_main_mpi/num_iters * 1024.0d0**3), " GB/s"
        print '(A)', "-----------------------------------------------------"
    end if

#ifndef NO_GPU
    !$omp target exit data map(release: device_sendbuf, device_recvbuf)
#endif
    deallocate(device_sendbuf, device_recvbuf)
    call MPI_Finalize(ierr)

contains

    subroutine check_mpi_error(ierr, label)
        integer, intent(in) :: ierr
        character(len=*), intent(in) :: label
        character(len=MPI_MAX_ERROR_STRING) :: error_string
        integer :: resultlen, ignore

        if (ierr /= MPI_SUCCESS) then
            call MPI_Error_string(ierr, error_string, resultlen, ignore)
            print '(A, I0, A, A, A, A)', "ERROR on Rank ", rank, " at ", label, ": ", trim(error_string)
            call MPI_Abort(MPI_COMM_WORLD, ierr, ignore)
        end if
    end subroutine check_mpi_error

end program mpi_omp_gpu_benchmark
