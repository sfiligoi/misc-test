!
! Acknowledgement:
!  Base program genergated with Google Gemini
!  Some additional manual changes after that
!
program mpi_omp_gpu_allreduce
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
    character(len=32) :: arg_str
    real(8) :: recvmax
    
    ! Timers
    real(8) :: t0, t1
    real(8) :: t_warm_comp, t_warm_mpi, t_warm_total
    real(8) :: t_main_comp, t_main_mpi, t_main_total
    
    ! MPI_Allreduce needs a send and a receive buffer
    real(8), dimension(:), allocatable :: device_sendbuf, device_recvbuf

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

    ! 1. CLI Arguments (Count is the number of REAL8 elements)
    arg_count = command_argument_count()
    if (arg_count < 2) then
        if (rank == 0) print *, "Usage: mpirun ./mpi_omp_gpu_allreduce <count> <iterations>"
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    call get_command_argument(1, arg_str); read(arg_str, *) count
    call get_command_argument(2, arg_str); read(arg_str, *) num_iters

    allocate(device_sendbuf(count), device_recvbuf(count))

    ! Initial GPU Map and Data Setup
#ifndef NO_GPU
    !$omp target enter data map(alloc: device_sendbuf, device_recvbuf)
    !$omp target teams distribute parallel do
#else
    !$omp parallel do
#endif
    do i = 1, count
        device_sendbuf(i) = 1.0d0+0.0001d0*i
        device_recvbuf(i) = 0.0d0
    end do

    t_warm_comp = 0.0; t_warm_mpi = 0.0; t_main_comp = 0.0; t_main_mpi = 0.0

    ! --- 2. TIMED WARMUP PHASE ---
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
        do i = 1, count
            device_sendbuf(i) = device_sendbuf(i) * 1.000001d0 + device_recvbuf(i)/count
        end do
        t1 = MPI_Wtime(); t_warm_comp = t_warm_comp + (t1 - t0)

        t0 = t1
#ifndef NO_GPU
        !$omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
            call MPI_Allreduce(device_sendbuf, device_recvbuf, count, &
                               MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
#ifndef NO_GPU
        !$omp end target data
#endif
        t1 = MPI_Wtime(); t_warm_mpi = t_warm_mpi + (t1 - t0)
        call check_mpi_error(ierr, "Warmup AllReduce")
    end do
    t_warm_total = MPI_Wtime() - t_warm_total

    ! Basic check that recvbuffer is being updated in GPU memory
    recvmax = -1
#ifndef NO_GPU
    !$omp target teams distribute parallel do reduction(max:recvmax) map(tofrom:recvmax)
#endif
    do i = 1, count
        if (recvmax< device_recvbuf(i)) recvmax=device_recvbuf(i)
    end do

    if (recvmax<1) then
        print '(A, F20.15, A, I0)', "Recv error in warmup, >0 expected, found ", recvmax, " rank= ", rank
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    endif

    ! --- 3. TIMED MAIN PHASE ---
    if (rank == 0) print '(A, I0, A)', ">>> Starting Main Loop (", num_iters, " iters)"
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    call check_mpi_error(ierr, "Main Barrier")
    t_main_total = MPI_Wtime()

    do iter = 1, num_iters
        t0 = MPI_Wtime()
        ! Mimic compute: simple floating point operation on GPU
#ifndef NO_GPU
        !$omp target teams distribute parallel do
#else
        !$omp parallel do
#endif
        do i = 1, count
            device_sendbuf(i) = device_sendbuf(i) * 1.000001d0 + device_recvbuf(i)/count
        end do
        t1 = MPI_Wtime(); t_main_comp = t_main_comp + (t1 - t0)

        t0 = t1
        ! Use device pointers for GPU-aware reduction
#ifndef NO_GPU
        !$omp target data use_device_ptr(device_sendbuf, device_recvbuf)
#endif
            call MPI_Allreduce(device_sendbuf, device_recvbuf, count, &
                               MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
#ifndef NO_GPU
        !$omp end target data
#endif
        t1 = MPI_Wtime(); t_main_mpi = t_main_mpi + (t1 - t0)
        call check_mpi_error(ierr, "Main AllReduce")
    end do
    t_main_total = MPI_Wtime() - t_main_total

    ! --- 4. REPORTING ---
    if (rank == (nranks-1)) then
        print '(/, A)', "---------------- ALLREDUCE PERFORMANCE ----------------"
        print '(A, I0, A, I0)', " Ranks: ", nranks, " | Count: ", count
        print '(A, F12.6, A)',  " Main Total:   ", t_main_total, " s"
        print '(/, A)', " AVERAGE TIME PER ITERATION"
        print '(A, F12.6, A, F12.6, A)', " Compute: ", t_main_comp/num_iters, " s  (Warmup: ", t_warm_comp/warmup_iters, " s)"
        print '(A, F12.6, A, F12.6, A)', " MPI:     ", t_main_mpi/num_iters,  " s  (Warmup: ", t_warm_mpi/warmup_iters,  " s)"
        
        ! Algorithm for AllReduce usually involves 2 * log2(P) or 2 * (P-1)/P data movement
        ! Here we report simple throughput: (count * bytes) / time
        print '(/, A, F10.2, A)', " Avg MPI Throughput: ", &
            (real(count,8) * 8.0d0) / (t_main_mpi/num_iters * 1024.0d0**3), " GB/s"
        print '(A)', "-------------------------------------------------------"
    end if

    ! Final check to ensure MPI worked
#ifndef NO_GPU
    !$omp target update from(device_sendbuf(1:3))
    !$omp target update from(device_recvbuf(1:3))
#endif
    ! Value will be proporitonal to both count and n_rank
    if (rank == (nranks-1)) print *, "Sample Send Buffer elements (Index 1-3): ", device_sendbuf(1:3)
    if (rank == (nranks-1)) print *, "Sample Revc Buffer elements (Index 1-3): ", device_recvbuf(1:3)

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

end program mpi_omp_gpu_allreduce
