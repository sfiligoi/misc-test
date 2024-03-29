program fortran_test_fft

#ifdef _OPENACC
  
#ifdef HIPGPU
  ! HIP
#define USEHIPFFT 1
#else
  ! CUDA
#define USECUFFT 1
#endif

#else
  ! FFTW
#define USEFFTW 1
#endif

  implicit none

  CHARACTER(len=32) :: arg
  integer :: d1,d2,nbatch

  if (COMMAND_ARGUMENT_COUNT()/=3) then
     write(*,*) "Usage:"
     write(*,*) "  fortran_test_fft d1 d2 batch"
     call exit(1)
  endif

  CALL get_command_argument(1, arg)
  read(arg,*) d1
  CALL get_command_argument(2, arg)
  read(arg,*) d2
  CALL get_command_argument(3, arg)
  read(arg,*) nbatch

  write(*,*) "Processing 2D FFTs (",d1,"x",d2,") in batches of ",nbatch

  ! switch d1 and d2 due to FORTRAN memory layout
  call test_fft(d2,d1,nbatch)

contains
  subroutine test_fft(d1,d2,nbatch)

  use, intrinsic :: iso_c_binding
  implicit none
  !-----------------------------------
   integer, intent(in) :: d1,d2,nbatch

#ifdef USEHIPFFT
  ! HIP
  type(C_PTR) :: plan_c2r_many
  type(C_PTR) :: plan_r2c_many
#endif
#ifdef USECUFFT
  ! CUDA
  integer(c_int) :: plan_c2r_many
  integer(c_int) :: plan_r2c_many
#endif
#ifdef USEFFTW
  ! FFTW
  type(C_PTR) :: plan_c2r_many
  type(C_PTR) :: plan_r2c_many
#endif

  complex, dimension(:,:,:), allocatable :: fxmany,fvmany
  real, dimension(:,:,:), allocatable :: uxmany,uvmany
  integer :: i,j,k
  integer :: start_count, end_count 
  integer :: count_rate, count_max

  allocate( fxmany(0:d1/2,0:d2-1,nbatch) )
  allocate( fvmany(0:d1/2,0:d2-1,nbatch) )
  allocate( uxmany(0:d1-1,0:d2-1,nbatch) )
  allocate( uvmany(0:d1-1,0:d2-1,nbatch) )


  fxmany = 0.0
  fvmany = 0.0
  uxmany = 0.0
  uvmany = 0.0
#ifdef USEOPENMPOFFLOAD
!$omp target enter data map(to:fxmany,fvmany,uxmany,uvmany)
#else
!$acc enter data copyin( fxmany,fvmany,uxmany,uvmany)
#endif

  ! semi-arbirtrary initialization to non-zero
#ifdef USEOPENMPOFFLOAD
!$omp target teams loop collapse(2) private(i,j,k)
#else
!$acc parallel loop collapse(2) private(i,j,k)
#endif
  do k=1,nbatch
   do j=0,d2-1
     do i=max(0,j-4),min(j+4,d1/2)
        fxmany(i,j,k) = cmplx(0.33/(i+1) + 0.22/(j+i) + 0.01/k, 0.21/(i+1) + 0.34/(j+i) + 0.12/k)
     enddo
     do i=max(0,j-4),min(j+4,d1-1)
        uvmany(i,j,k) = sqrt(0.44/(i+1) + 0.29/(j+i) + 0.11/k)
     enddo
   enddo
  enddo
#ifdef USEOPENMPOFFLOAD
!$omp barrier
#else
!$acc wait
#endif

  call test_setup_fft(d1,d2,nbatch,plan_c2r_many,plan_r2c_many,fxmany,uvmany,uxmany,fvmany)

  call test_do_fft(plan_c2r_many,plan_r2c_many, fxmany,uvmany,uxmany,fvmany)

#ifdef USEOPENMPOFFLOAD
!$omp barrier
#else
!$acc wait
#endif
  do j=1,5
   call SYSTEM_CLOCK(start_count, count_rate, count_max)
   do i=1,100
    call test_do_fft(plan_c2r_many,plan_r2c_many,fxmany,uvmany,uxmany,fvmany)
#ifdef USEOPENMPOFFLOAD
!$omp barrier
#else
!$acc wait
#endif
   enddo
   call SYSTEM_CLOCK(end_count, count_rate, count_max)

   if ((end_count<start_count).and.(count_max>0)) end_count = end_count + count_max
   write(*,*) "4x100 FFT took ", (1.0*(end_count-start_count))/count_rate, " seconds"
  enddo

#ifdef USEOPENMPOFFLOAD
!$omp target exit data map(delete:fxmany,fvmany,uxmany,uvmany)
#else
!$acc exit data delete( fxmany,fvmany,uxmany,uvmany)
#endif
  end subroutine test_fft

  subroutine test_setup_fft(d1,d2,nbatch,plan_c2r_many,plan_r2c_many,fxmany,uvmany,uxmany,fvmany)

  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env
#ifdef USEHIPFFT
  use hipfort_hipfft
#endif
#ifdef USECUFFT
  use cufft
#endif

  implicit none
#ifdef USEFFTW
     include 'fftw3.f03'
#endif
#ifdef _OPENMP
     integer, external :: omp_get_max_threads
#endif

  !-----------------------------------
   integer, intent(in) :: d1,d2,nbatch

#ifdef USEHIPFFT
  type(C_PTR), intent(inout) :: plan_c2r_many
  type(C_PTR), intent(inout) :: plan_r2c_many
#endif
#ifdef USECUFFT
  integer(c_int), intent(inout) :: plan_c2r_many
  integer(c_int), intent(inout) :: plan_r2c_many
#endif
#ifdef USEFFTW
  ! FFTW
  type(C_PTR), intent(inout) :: plan_c2r_many
  type(C_PTR), intent(inout) :: plan_r2c_many
#endif

  complex, dimension(:,:,:), intent(inout) :: fxmany
  real, dimension(:,:,:), intent(inout) :: uvmany
  real, dimension(:,:,:), intent(inout) :: uxmany
  complex, dimension(:,:,:), intent(inout) :: fvmany
  !-----------------------------------
  integer :: istatus
  integer, parameter :: irank = 2
  integer, dimension(irank) :: ndim,inembed,onembed
  integer :: idist,odist,istride,ostride,nsplit

#ifdef USEFFTW
#ifdef _OPENMP
     istatus = fftw_init_threads()
     call fftw_plan_with_nthreads(omp_get_max_threads())
#endif
#endif

     ndim(1) = d2
     ndim(2) = d1
     idist = size(fxmany,1)*size(fxmany,2)
     odist = size(uxmany,1)*size(uxmany,2)
     istride = 1
     ostride = 1
     inembed = size(fxmany,1)
     onembed = size(uxmany,1)

#ifdef USEHIPFFT
     plan_c2r_many = c_null_ptr
     istatus = hipfftPlanMany(&
          plan_c2r_many, &
          irank, &
          ndim, &
          inembed, &
          istride, &
          idist, &
          onembed, &
          ostride, &
          odist, &
          HIPFFT_Z2D, &
          nbatch)
#endif
#ifdef USECUFFT
     istatus = cufftPlanMany(&
          plan_c2r_many, &
          irank, &
          ndim, &
          inembed, &
          istride, &
          idist, &
          onembed, &
          ostride, &
          odist, &
          CUFFT_Z2D, &
          nbatch)
#endif
#ifdef USEFFTW
     plan_c2r_many = fftw_plan_many_dft_c2r(&
          irank, &
          ndim, &
          nbatch, &
          fxmany, &
          inembed, &
          istride, &
          idist, &
          uxmany, &
          onembed, &
          ostride, &
          odist, &
          FFTW_MEASURE)
     istatus = 0
#endif
#ifdef USEOPENMPOFFLOAD
!$omp barrier
#else
!$acc wait
#endif
     if (istatus/=0) then
        write(*,*) "ERROR: fftPlanMany Z2D failed! ", istatus
        call abort
     else
        write(*,*) "INFO: fftPlanMany Z2D done."
        call flush(6)
     endif

     idist = size(uxmany,1)*size(uxmany,2)
     odist = size(fxmany,1)*size(fxmany,2)
     inembed = size(uxmany,1)
     onembed = size(fxmany,1)
     istride = 1
     ostride = 1

#ifdef USEHIPFFT
     plan_r2c_many = c_null_ptr
     istatus = hipfftPlanMany(&
          plan_r2c_many, &
          irank, &
          ndim, &
          inembed, &
          istride, &
          idist, &
          onembed, &
          ostride, &
          odist, &
          HIPFFT_D2Z, &
          nbatch)
#endif
#ifdef USECUFFT
     istatus = cufftPlanMany(&
          plan_r2c_many, &
          irank, &
          ndim, &
          inembed, &
          istride, &
          idist, &
          onembed, &
          ostride, &
          odist, &
          CUFFT_D2Z, &
          nbatch)
#endif
#ifdef USEFFTW
     plan_r2c_many = fftw_plan_many_dft_r2c(&
          irank, &
          ndim, &
          nbatch, &
          uvmany, &
          inembed, &
          istride, &
          idist, &
          fvmany, &
          onembed, &
          ostride, &
          odist, &
          FFTW_MEASURE)
     istatus = 0
#endif
#ifdef USEOPENMPOFFLOAD
!$omp barrier
#else
!$acc wait
#endif
     if (istatus/=0) then
        write(*,*) "ERROR: fftPlanMany D2Z failed! ", istatus
        call abort
     else
        write(*,*) "INFO: fftPlanMany D2Z done."
        call flush(6)
     endif

  end subroutine test_setup_fft

  subroutine test_do_fft(plan_c2r_many,plan_r2c_many,fxmany,uvmany,uxmany,fvmany)

     use, intrinsic :: iso_c_binding
#ifdef USEHIPFFT
  use hipfort_hipfft
#endif
#ifdef USECUFFT
  use cufft
#endif

  implicit none
#ifdef USEFFTW
     include 'fftw3.f03'
#endif
#ifdef USEHIPFFT
     type(C_PTR), intent(inout) :: plan_c2r_many
     type(C_PTR), intent(inout) :: plan_r2c_many
#endif
#ifdef USECUFFT
     integer(c_int), intent(inout) :: plan_c2r_many
     integer(c_int), intent(inout) :: plan_r2c_many
#endif
#ifdef USEFFTW
     type(C_PTR), intent(inout) :: plan_c2r_many
     type(C_PTR), intent(inout) :: plan_r2c_many
#endif
     complex, dimension(:,:,:), intent(inout) :: fxmany
     real, dimension(:,:,:), intent(inout) :: uvmany
     real, dimension(:,:,:), intent(inout) :: uxmany
     complex, dimension(:,:,:), intent(inout) :: fvmany
     !-----------------------------------
     integer :: rc

     ! --------------------------------------
     ! Forward
     ! --------------------------------------
#ifdef USEOPENMPOFFLOAD
!$omp  target data use_device_ptr(fxmany,uxmany) 
#else
!$acc  host_data use_device(fxmany,uxmany) 
#endif

#ifdef USEHIPFFT
     rc = hipfftExecZ2D(plan_c2r_many,c_loc(fxmany),c_loc(uxmany))
#endif
#ifdef USECUFFT
     rc = cufftExecZ2D(plan_c2r_many,fxmany,uxmany)
#endif
#ifdef USEFFTW
     call fftw_execute_dft_c2r(plan_c2r_many,fxmany,uxmany) 
     rc = 0
#endif
#ifdef USEOPENMPOFFLOAD
!$omp barrier
!$omp end target data
#else
!$acc wait
!$acc end host_data
#endif

     if (rc/=0) then
        write(*,*) "ERROR: fftExec D2Z failed! ", rc
        call abort
#ifdef DEBUG
     else
        write(*,*) "INFO: fftExec D2Z done."
        call flush(6)
#endif
     endif


     ! --------------------------------------
     ! Backward
     ! --------------------------------------

#ifdef USEOPENMPOFFLOAD
!$omp  target data use_device_ptr(uvmany,fvmany) 
#else
!$acc host_data use_device(uvmany,fvmany)
#endif

#ifdef USEHIPFFT
     rc = hipfftExecD2Z(plan_r2c_many,c_loc(uvmany),c_loc(fvmany))
#endif
#ifdef USECUFFT
     rc = cufftExecD2Z(plan_r2c_many,uvmany,fvmany)
#endif
#ifdef USEFFTW
     call fftw_execute_dft_r2c(plan_r2c_many,uvmany,fvmany) 
     rc = 0
#endif

#ifdef USEOPENMPOFFLOAD
!$omp barrier
!$omp end target data
#else
!$acc wait
!$acc end host_data
#endif

     if (rc/=0) then
        write(*,*) "ERROR: fftExec Z2D failed! ", rc
        call abort
#ifdef DEBUG
     else
        write(*,*) "INFO: fftExec Z2D done."
        call flush(6)
#endif
     endif

     ! --------------------------------------
     ! Backward2 - use different buffer
     ! --------------------------------------

#ifdef USEOPENMPOFFLOAD
!$omp  target data use_device_ptr(uxmany,fvmany) 
#else
!$acc host_data use_device(uxmany,fvmany)
#endif

#ifdef USEHIPFFT
     rc = hipfftExecD2Z(plan_r2c_many,c_loc(uxmany),c_loc(fvmany))
#endif
#ifdef USECUFFT
     rc = cufftExecD2Z(plan_r2c_many,uxmany,fvmany)
#endif
#ifdef USEFFTW
     call fftw_execute_dft_r2c(plan_r2c_many,uxmany,fvmany) 
     rc = 0
#endif

#ifdef USEOPENMPOFFLOAD
!$omp barrier
!$omp end target data
#else
!$acc wait
!$acc end host_data
#endif
     if (rc/=0) then
        write(*,*) "ERROR: fftExec Z2D failed! ", rc
        call abort
#ifdef DEBUG
     else
        write(*,*) "INFO: fftExec Z2D done."
        call flush(6)
#endif
     endif

     ! --------------------------------------
     ! Forward2 - use different buffer
     ! --------------------------------------
#ifdef USEOPENMPOFFLOAD
!$omp  target data use_device_ptr(fvmany,uxmany) 
#else
!$acc  host_data use_device(fvmany,uxmany) 
#endif

#ifdef USEHIPFFT
     rc = hipfftExecZ2D(plan_c2r_many,c_loc(fvmany),c_loc(uxmany))
#endif
#ifdef USECUFFT
     rc = cufftExecZ2D(plan_c2r_many,fvmany,uxmany)
#endif
#ifdef USEFFTW
     ! FFTW
     call fftw_execute_dft_c2r(plan_c2r_many,fvmany,uxmany) 
     rc = 0
#endif

#ifdef USEOPENMPOFFLOAD
!$omp barrier
!$omp end target data
#else
!$acc wait
!$acc end host_data
#endif

     if (rc/=0) then
        write(*,*) "ERROR: fftExec D2Z failed! ", rc
        call abort
#ifdef DEBUG
     else
        write(*,*) "INFO: fftExec D2Z done."
        call flush(6)
#endif
     endif

  end subroutine test_do_fft


end program fortran_test_fft
