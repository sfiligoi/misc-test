program fortran_test_fft


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

#ifdef _OPENACC
  
#ifdef HIPGPU
  ! HIP
  type(C_PTR) :: plan_c2r_many
  type(C_PTR) :: plan_r2c_many
#else
  ! CUDA
  integer(c_int) :: plan_c2r_many
  integer(c_int) :: plan_r2c_many
#endif

#else
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
!$acc enter data copyin( fxmany,fvmany,uxmany,uvmany)

  ! semi-arbirtrary initialization to non-zero
!$acc parallel loop collapse(2)
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

  call test_setup_fft(d1,d2,nbatch,plan_c2r_many,plan_r2c_many,fxmany,uvmany,uxmany,fvmany)

  call test_do_fft(plan_c2r_many,plan_r2c_many, fxmany,uvmany,uxmany,fvmany)

  do j=1,5
   call SYSTEM_CLOCK(start_count, count_rate, count_max)
   do i=1,100
    call test_do_fft(plan_c2r_many,plan_r2c_many,fxmany,uvmany,uxmany,fvmany)
   enddo
   call SYSTEM_CLOCK(end_count, count_rate, count_max)

   if ((end_count<start_count).and.(count_max>0)) end_count = end_count + count_max
   write(*,*) "4x100 FFT took ", (1.0*(end_count-start_count))/count_rate, " seconds"
  enddo

!$acc exit data delete( fxmany,fvmany,uxmany,uvmany)
  end subroutine test_fft

  subroutine test_setup_fft(d1,d2,nbatch,plan_c2r_many,plan_r2c_many,fxmany,uvmany,uxmany,fvmany)

  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env
#ifdef _OPENACC
#ifdef HIPGPU
  use hipfort_hipfft
#else
  use cufft
#endif
#endif
  implicit none
#ifndef _OPENACC
     include 'fftw3.f03'
#ifdef _OPENACC
     integer, external :: omp_get_max_threads
#endif
#endif
  !-----------------------------------
   integer, intent(in) :: d1,d2,nbatch
#ifdef _OPENACC
  
#ifdef HIPGPU
  type(C_PTR), intent(inout) :: plan_c2r_many
  type(C_PTR), intent(inout) :: plan_r2c_many
#else
  integer(c_int), intent(inout) :: plan_c2r_many
  integer(c_int), intent(inout) :: plan_r2c_many
#endif

#else
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

#ifndef _OPENACC
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

#ifdef _OPENACC
  
#ifdef HIPGPU
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
#else
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

#else
  ! FFTW
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

#ifdef _OPENACC
  
#ifdef HIPGPU
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
#else
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

#else
  ! FFTW
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
#ifdef _OPENACC
#ifdef HIPGPU
     use hipfort_hipfft
#else
     use cufft
#endif
#endif

     implicit none
     !-----------------------------------
#ifndef _OPENACC
     include 'fftw3.f03'
#endif

#ifdef _OPENACC
  
#ifdef HIPGPU
     type(C_PTR), intent(inout) :: plan_c2r_many
     type(C_PTR), intent(inout) :: plan_r2c_many
#else
     integer(c_int), intent(inout) :: plan_c2r_many
     integer(c_int), intent(inout) :: plan_r2c_many
#endif

#else
     ! FFTW
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
!$acc  host_data &
!$acc& use_device(fxmany,uxmany) 

#ifdef _OPENACC
  
#ifdef HIPGPU
     rc = hipfftExecZ2D(plan_c2r_many,c_loc(fxmany),c_loc(uxmany))
#else
     rc = cufftExecZ2D(plan_c2r_many,fxmany,uxmany)
#endif

#else
     ! FFTW
     call fftw_execute_dft_c2r(plan_c2r_many,fxmany,uxmany) 
     rc = 0
#endif
!$acc wait
!$acc end host_data

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

!$acc host_data use_device(uvmany,fvmany)
#ifdef _OPENACC
  
#ifdef HIPGPU
     rc = hipfftExecD2Z(plan_r2c_many,c_loc(uvmany),c_loc(fvmany))
#else
     rc = cufftExecD2Z(plan_r2c_many,uvmany,fvmany)
#endif

#else
     ! FFTW
     call fftw_execute_dft_r2c(plan_r2c_many,uvmany,fvmany) 
     rc = 0
#endif

!$acc wait
!$acc end host_data
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

!$acc host_data use_device(uxmany,fvmany)
#ifdef _OPENACC
  
#ifdef HIPGPU
     rc = hipfftExecD2Z(plan_r2c_many,c_loc(uxmany),c_loc(fvmany))
#else
     rc = cufftExecD2Z(plan_r2c_many,uxmany,fvmany)
#endif

#else
     ! FFTW
     call fftw_execute_dft_r2c(plan_r2c_many,uxmany,fvmany) 
     rc = 0
#endif

!$acc wait
!$acc end host_data
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
!$acc  host_data use_device(fvmany,uxmany) 

#ifdef _OPENACC
  
#ifdef HIPGPU
     rc = hipfftExecZ2D(plan_c2r_many,c_loc(fvmany),c_loc(uxmany))
#else
     rc = cufftExecZ2D(plan_c2r_many,fvmany,uxmany)
#endif

#else
     ! FFTW
     call fftw_execute_dft_c2r(plan_c2r_many,fvmany,uxmany) 
     rc = 0
#endif
!$acc wait
!$acc end host_data

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
