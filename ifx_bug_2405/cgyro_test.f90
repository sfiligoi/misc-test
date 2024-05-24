program cgyro

  use cgyro_globals

  delta_t=1.0e-6

  nc=96
  nt1=0
  nt2=0
  allocate(field_old(2,nc,nt1:nt2))
  allocate(field_old2(2,nc,nt1:nt2))
  allocate(freq(nt1:nt2))

  field_old = 1
  field_old2 = 2

  i_time = 0
  call cgyro_freq
  i_time = 1
  call cgyro_freq

end program cgyro

