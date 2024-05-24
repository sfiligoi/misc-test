!---------------------------------------------------------
! cgyro_freq.f90
!
! PURPOSE:
!  Compute estimates of linear eigenvalue w where:
!
!                   -iwt
!            phi ~ e
!---------------------------------------------------------

subroutine cgyro_freq

  use cgyro_globals

  implicit none

  real :: mw
  real :: total_weight,dfr,dfi
  real, dimension(nc) :: mode_weight
  integer :: itor
  complex, dimension(nc) :: freq_loc
  complex :: fl,myfr,df,total_weighted_freq
  complex :: fo1,fo2
  complex :: icdt

  if (i_time == 0) then

    freq(:) = 0.0

  else

    !--------------------------------------------------
    ! Standard method: sum all wavenumbers at a given n
    !--------------------------------------------------

    icdt = i_c/delta_t
    do itor=nt1,nt2

     total_weight = 0.0
     total_weighted_freq = (0.0,0.0)
     do ic=1,nc
        ! Use potential to compute frequency
        fo1 = field_old(1,ic,itor)
        fo2 = field_old2(1,ic,itor)
        mw = abs(fo1)
        mode_weight(ic) =  mw
        total_weight = total_weight + mw
        ! Define local frequencies
        if ( (mw > 1e-12) .and. (abs(fo2) > 1e-12) ) then
           fl = icdt*log(fo1/fo2)
        else
           fl = 0.0
        endif
        freq_loc(ic) = fl
        total_weighted_freq = total_weighted_freq + fl*mw
     enddo

     myfr = total_weighted_freq/total_weight
     write(*,*) myfr, total_weighted_freq, total_weight
     freq(itor) = myfr
    
    enddo

  endif


end subroutine cgyro_freq
