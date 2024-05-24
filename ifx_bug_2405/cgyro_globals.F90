!-----------------------------------------------------------------
! cgyro_globals.f90
!
! PURPOSE:
!  CGYRO global variables.  The idea is to have a primary, large
!  module containing all essential CGYRO arrays and scalars.
!-----------------------------------------------------------------

module cgyro_globals

  use, intrinsic :: iso_c_binding
#if defined(_OPENACC) || defined(OMPGPU)
#define CGYRO_GPU_FFT
#endif

  use, intrinsic :: iso_fortran_env
  
  ! Data output precision setting
  integer, parameter :: BYTE=4 ! Change to 8 for double precision
  
  !---------------------------------------------------------------
  ! Input parameters:
  !
  real    :: delta_t

  !
  ! Pointers
  integer :: nc,ic
  integer :: nc_loc,ic_loc
  !
  integer :: nt1,nt2,nt_loc
  !---------------------------------------------------------------

  !---------------------------------------------------------------
  ! Constants
  !
  complex, parameter :: i_c  = (0.0,1.0)
  !---------------------------------------------------------------

  integer :: i_time
  complex, dimension(:), allocatable :: freq
  complex :: freq_err
  !---------------------------------------------------------------

  !
  ! Fields
  complex, dimension(:,:,:), allocatable :: field_old
  complex, dimension(:,:,:), allocatable :: field_old2

end module cgyro_globals
