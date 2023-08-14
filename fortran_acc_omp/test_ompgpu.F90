program test_ompgpu

  implicit none

  real, dimension(0:189,0:767,72)  :: uvmany
  real, dimension(0:189,0:767,72) :: comp_uxmany
  integer :: i,j,k,l
  real :: utmp
  integer :: start_count, end_count 
  integer :: count_rate, count_max

#if defined(OMPGPU)
!$omp target enter data map(to:uvmany)
!$omp target enter data map(to:comp_uxmany)
#elif defined(_OPENACC)
!$acc enter data create(uvmany)
!$acc enter data create(comp_uxmany)
#endif

  do i=1,72
    do j=0,767
      do k=0,189
        uvmany(k,j,i) = 1.0+ 0.001*i + 1.e-5*k + 1.e-9*j
        comp_uxmany(k,j,i) = 1.0+ 0.001*k + 1.e-5*j + 1.e-9*i
      enddo
    enddo
  enddo

#if defined(OMPGPU)
!$omp target update to(uvmany)
!$omp target update to(comp_uxmany)
#else
!$acc update device(uvmany)
!$acc update device(comp_uxmany)
#endif

  utmp = sum(comp_uxmany(:,:,:))
  write(*,*) "comp result ", utmp, " should be ~11500000"

  ! do some pure compute benchmarking 
  call SYSTEM_CLOCK(start_count, count_rate, count_max)
#ifdef OMPGPU
!$omp target teams distribute parallel do collapse(3) private(l,utmp)
#else
!$acc parallel loop gang vector collapse(3) private(l,utmp) present(comp_uxmany,uvmany)
#endif
  do i=1,72
    do j=0,767
      do k=0,189
        utmp = 0
!$acc loop seq
        do l=0,10000
          utmp = utmp + ((9.e-7*l)*uvmany(modulo((i+k+l),190),j,i))**2
        enddo
        comp_uxmany(k,j,i) = utmp
      enddo
    enddo
  enddo
  call SYSTEM_CLOCK(end_count, count_rate, count_max)

  if ((end_count<start_count).and.(count_max>0)) end_count = end_count + count_max
  write(*,*) "basic bench took ", (1.0*(end_count-start_count))/count_rate, " seconds"

#if defined(OMPGPU)
!$omp target update from(comp_uxmany)
#else
!$acc update host(comp_uxmany)
#endif

  utmp = sum(comp_uxmany(:,:,:))
  write(*,*) "comp result ", utmp, " should be ~3050000"

  ! do some pure compute benchmarking, a variant 
  call SYSTEM_CLOCK(start_count, count_rate, count_max)
#ifdef OMPGPU
!$omp target teams distribute collapse(2)
#else
!$acc parallel loop gang collapse(2) present(comp_uxmany,uvmany)
#endif
  do i=1,72
    do j=0,767
#ifdef OMPGPU
!$omp parallel do private(l,utmp)
#else
!$acc loop vector private(l,utmp)
#endif
      do k=0,189
        utmp = 0
        do l=0,10000
          utmp = utmp + ((7.e-7*l)*uvmany(modulo((i+k+l),190),j,i))**2
        enddo
        comp_uxmany(k,j,i) = utmp
      enddo
    enddo
  enddo
  call SYSTEM_CLOCK(end_count, count_rate, count_max)

  if ((end_count<start_count).and.(count_max>0)) end_count = end_count + count_max
  write(*,*) "variant bench took ", (1.0*(end_count-start_count))/count_rate, " seconds"

#if defined(OMPGPU)
!$omp target update from(comp_uxmany)
#else
!$acc update host(comp_uxmany)
#endif

  utmp = sum(comp_uxmany(:,:,:))
  write(*,*) "comp result ", utmp, " should be ~1840000"

  ! do some slightly more complex compute benchmarking 
  call SYSTEM_CLOCK(start_count, count_rate, count_max)
#ifdef OMPGPU
!$omp target teams
!$omp distribute collapse(2) private(l,utmp)
#else
!$acc parallel loop gang collapse(2) private(l,utmp) present(comp_uxmany,uvmany)
#endif
  do i=1,72
    do j=0,766,2

#ifdef OMPGPU
!$omp parallel do simd private(l,utmp)
#else
!$acc loop vector private(l,utmp)
#endif
      do k=0,189
        utmp = 0
!$acc loop seq
        do l=0,10000
          utmp = utmp + ((9.e-7*l)*uvmany(modulo((i+k+l),190),j,i))**2
        enddo
        comp_uxmany(k,j,i) = utmp
      enddo

#ifdef OMPGPU
!$omp parallel do simd private(l,utmp)
#else
!$acc loop vector private(l,utmp)
#endif
      do k=0,189
        utmp = 0
!$acc loop seq
        do l=0,10000
          utmp = utmp + ((9.e-7*l)*comp_uxmany(modulo((i+k+l),190),j,i))**2
        enddo
        uvmany(k,j,i) = utmp
      enddo

    enddo
  enddo
#ifdef OMPGPU
!$omp end target teams
#endif
  call SYSTEM_CLOCK(end_count, count_rate, count_max)

  if ((end_count<start_count).and.(count_max>0)) end_count = end_count + count_max
  write(*,*) "2 loop bench took ", (1.0*(end_count-start_count))/count_rate, " seconds"

#if defined(OMPGPU)
!$omp target update from(uvmany)
#else
!$acc update host(uvmany)
#endif

  utmp = sum(uvmany(:,:,:))
  write(*,*) "comp result ", utmp, " should be ~5569000"

  ! test dependencies slightly more complex compute benchmarking 
  call SYSTEM_CLOCK(start_count, count_rate, count_max)
#ifdef OMPGPU
!$omp target teams distribute parallel do collapse(3) private(l,utmp) &
!$omp& depend(out:comp_uxmany(:,:,:)) nowait
#else
!$acc parallel loop gang vector collapse(3) async(1) &
!$acc& private(l,utmp) present(comp_uxmany,uvmany)
#endif
  do i=1,72
    do j=0,767
      do k=0,189
        utmp = 0
!$acc loop seq
        do l=0,10000
          utmp = utmp + ((9.e-7*l)*uvmany(modulo((i+k+l),190),j,i))**2
        enddo
        comp_uxmany(k,j,i) = utmp
      enddo
    enddo
  enddo

#ifdef OMPGPU
!$omp target teams distribute parallel do collapse(3) private(l,utmp) &
!$omp& depend(in:comp_uxmany(:,:,:)) depend(out:uvmany(:,:,:)) nowait
#else
!$acc parallel loop gang vector collapse(3) wait(1) async(1) &
!$acc& private(l,utmp) present(comp_uxmany,uvmany)
#endif
  do i=1,72
    do j=0,767
      do k=0,189
        utmp = 0
!$acc loop seq
        do l=0,10000
          utmp = utmp + ((1.e-7*l)*comp_uxmany(modulo((i+k+l),190),j,i))**2
        enddo
        uvmany(k,j,i) = utmp
      enddo
    enddo
  enddo
#ifdef OMPGPU
!$omp target update from(uvmany) depend(in:uvmany(:,:,:))
#else
!$acc update host(uvmany) wait(1)
#endif
  call SYSTEM_CLOCK(end_count, count_rate, count_max)

  if ((end_count<start_count).and.(count_max>0)) end_count = end_count + count_max
  write(*,*) "dep loop bench took ", (1.0*(end_count-start_count))/count_rate, " seconds"

  utmp = sum(uvmany(:,:,:))
  write(*,*) "comp result ", utmp, " should be ~1482"

end program test_ompgpu
