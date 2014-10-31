! quicklens/shts/shts.f95
! --
! this file contains Fortran code implementing the spherical harmonic
! recursion relations described in quicklens/notes/shts, and routines
! to convert between vlm harmonic coefficients and complex maps.
!
! NOTE: THESE ROUTINES ARE ONLY WRITTEN FOR SPIN s>=0

subroutine vlm2map(ntht, nphi, lmax, s, tht, phi, vlm, map)
  integer ntht, nphi, lmax, s
  double precision tht(ntht), phi(nphi)
  double complex, intent(in)  :: vlm(0:((lmax+1)*(lmax+1)-1))
  double complex, intent(out), dimension(ntht, nphi) :: map
  double complex, save, allocatable ::  vtm(:,:)
  !f2py intent(in)   :: ntht, nphi, s, lmax
  !f2py intent(in)   :: tht, phi, vlm
  !f2py intent(hide) :: ntht, nphi
  !f2py intent(out)  :: map

  allocate( vtm(ntht,-lmax:lmax) )
  vtm(:,:) = 0.0
  map(:,:) = 0.0

  call vlm2vtm(ntht, lmax, s, tht, vlm, vtm)
  call vtm2map(ntht, nphi, lmax, phi, vtm, map)
  deallocate(vtm)
end subroutine vlm2map

subroutine vtm2map(ntht, nphi, lmax, phi, vtm, map)
  integer ntht, nphi, lmax
  double precision phi(nphi)
  double complex, intent(in)  :: vtm(ntht, -lmax:lmax)
  double complex, intent(out) :: map(ntht, nphi)

  integer p, m

  do p=1,nphi
     map(:,p) = vtm(:,0)
  end do

!$omp parallel do default(shared)
  do p=1,nphi
     do m=1,lmax
        map(:,p) = map(:,p) + vtm(:,+m) * &
             (cos(phi(p)*m)+(0.0,1.0)*sin(phi(p)*m)) + &
             vtm(:,-m) * (cos(phi(p)*m)-(0.0,1.0)*sin(phi(p)*m))
     end do
  end do
!$omp end parallel do
end subroutine vtm2map

subroutine vlm2vtm(ntht, lmax, s, tht, vlm, vtm)
  integer ntht, lmax, s
  double precision tht(ntht)
  double complex, intent(in)  :: vlm(0:((lmax+1)*(lmax+1)-1))
  double complex, intent(out) :: vtm(ntht, -lmax:lmax)

  integer l, m, tl, ts, tm, j
  double precision htttht(ntht), costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_m_lm0(ntht), llm_arr_m_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax), zl(0:lmax)

  sfac = 2.**40

  htttht(:)    = tan(tht(:)*0.5)
  costht(:)    = cos(tht(:))
  sintht(:)    = sin(tht(:))

  llm_arr_p(:) = 1./sqrt(8.*acos(0.0))
  llm_arr_m(:) = llm_arr_p(:)

  l = 0
  m = 0

  zl(:) = 0.0
  if (s .ne. 0) then
     do ts=1,s
        llm_arr_p(:) = llm_arr_p(:)*sqrt(1.+0.5/ts)*sintht(:)
     end do
     llm_arr_m(:) = llm_arr_p(:)
     l = s

     do tl=2,lmax
        zl(tl) = 1.0*s/(tl*(tl-1.))
     end do
  end if

  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.0

  vtm(:,0) = llm_arr_p_lm0(:)*vlm(l*l+l)

  rl(:) = 0.0
  do tl=l+1,lmax
     rl(tl) = sqrt( 1.0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.)) )
  end do

  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl) 

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

     vtm(:,0) = vtm(:,0) + llm_arr_p_lm0(:)*vlm(tl*tl+tl)
  end do

  spow_i(:) = 0.0

!$omp parallel do default(none) &
!$omp private(j, tl, tm, scal, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_m_lm0, llm_arr_m_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p, llm_arr_m) schedule(dynamic, 1) &
!$omp shared(htttht, costht, sintht, sfac, s, zl, ntht, tht, vlm, vtm)
  do tm=1,lmax
     do m=m+1,tm
        if (m<=s) then
           tfac = -sqrt( 1.0 * (s-m+1.) / (s+m) )
           llm_arr_p(:) = +llm_arr_p(:) * htttht(:) * tfac
           llm_arr_m(:) = +llm_arr_m(:) / htttht(:) * tfac
           l = s
        else
           tfac = +sqrt( 1.0 * m * (2.*m+1.)/(2.*(m+s)*(m-s)) )
           llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
           llm_arr_m(:) = +llm_arr_m(:) * tfac * sintht(:)
           l = m

           do j=1,ntht
              if (abs(llm_arr_p(j)) < 1./sfac) then
                 llm_arr_p(j) = llm_arr_p(j)*sfac
                 llm_arr_m(j) = llm_arr_m(j)*sfac
                 spow_i(j) = spow_i(j)-1
              end if
           end do
        end if
     end do
     m = tm

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.0

     llm_arr_m_lm0(:) = llm_arr_m(:)
     llm_arr_m_lm1(:) = 0.0

     vtm(:,+m) = llm_arr_p_lm0(:)*scal(:)*vlm(l*l+l+m)
     vtm(:,-m) = llm_arr_m_lm0(:)*scal(:)*vlm(l*l+l-m)

     rl(:) = 0.0
     do tl=l+1,lmax
        rl(tl) = sqrt( 1. * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.)) )
     end do

     do tl=l+1,lmax
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * (costht(:) + m*zl(tl)) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

        llm_arr_x_lmt(:) = (llm_arr_m_lm0(:) * (costht(:) - m*zl(tl)) - llm_arr_m_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_m_lm1(:) = llm_arr_m_lm0(:)
        llm_arr_m_lm0(:) = llm_arr_x_lmt(:)

        vtm(:,+m) = vtm(:,+m)+llm_arr_p_lm0(:)*scal(:)*vlm(tl*tl+tl+m)
        vtm(:,-m) = vtm(:,-m)+llm_arr_m_lm0(:)*scal(:)*vlm(tl*tl+tl-m)

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 llm_arr_m_lm0(j) = llm_arr_m_lm0(j)/sfac
                 llm_arr_m_lm1(j) = llm_arr_m_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine vlm2vtm

subroutine map2vlm(ntht, nphi, lmax, s, tht, phi, map, vlm)
  integer ntht, nphi, lmax, s
  double precision tht(ntht), phi(nphi)
  double complex, intent(in), dimension(ntht, nphi) :: map
  double complex, intent(out)  :: vlm(0:((lmax+1)*(lmax+1)-1))
  double complex, save, allocatable ::  vtm(:,:)
  !f2py intent(in)   :: ntht, nphi, s, lmax
  !f2py intent(in)   :: tht, phi, map
  !f2py intent(hide) :: ntht, nphi
  !f2py intent(out)  :: vlm

  allocate( vtm(ntht,-lmax:lmax) )
  vtm(:,:) = 0.0
  vlm(:)   = 0.0

  call map2vtm(ntht, nphi, lmax, phi, map, vtm)
  call vtm2vlm(ntht, lmax, s, tht, vtm, vlm)
  
  deallocate(vtm)
end subroutine map2vlm

subroutine map2vtm(ntht, nphi, lmax, phi, map, vtm)
  integer ntht, nphi, lmax
  double precision phi(nphi)
  double complex, intent(in)  :: map(ntht, nphi)
  double complex, intent(out) :: vtm(ntht, -lmax:lmax)

  integer p, m

  do p=1,nphi
     vtm(:,0) = vtm(:,0) + map(:,p)
  end do

!$omp parallel do default(shared)
  do p=1,nphi
     do m=1,lmax
        vtm(:,+m) = vtm(:,+m) + map(:,p) * &
             (cos(phi(p)*m)-(0.0,1.0)*sin(phi(p)*m))
        vtm(:,-m) = vtm(:,-m) + map(:,p) * &
             (cos(phi(p)*m)+(0.0,1.0)*sin(phi(p)*m))
     end do
  end do
!$omp end parallel do
end subroutine map2vtm

subroutine vtm2vlm(ntht, lmax, s, tht, vtm, vlm)
  integer ntht, lmax, s
  double precision tht(ntht)
  double complex, intent(in)  :: vtm(ntht, -lmax:lmax)
  double complex, intent(out) :: vlm(0:((lmax+1)*(lmax+1)-1))

  integer l, m, tl, ts, tm, j
  double precision htttht(ntht), costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_m_lm0(ntht), llm_arr_m_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax), zl(0:lmax)

  sfac = 2.**40

  htttht(:)    = tan(tht(:)*0.5)
  costht(:)    = cos(tht(:))
  sintht(:)    = sin(tht(:))

  llm_arr_p(:) = 1./sqrt(8.*acos(0.0))
  llm_arr_m(:) = llm_arr_p(:)

  l = 0
  m = 0

  zl(:) = 0.0
  if (s .ne. 0) then
     do ts=1,s
        llm_arr_p(:) = llm_arr_p(:)*sqrt(1.+0.5/ts)*sintht(:)
     end do
     llm_arr_m(:) = llm_arr_p(:)
     l = s

     do tl=2,lmax
        zl(tl) = 1.0*s/(tl*(tl-1.))
     end do
  end if

  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.0

  vlm(l*l+l) = sum( llm_arr_p_lm0(:)*vtm(:,0) )

  rl(:) = 0.0
  do tl=l+1,lmax
     rl(tl) = sqrt( 1.0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.)) )
  end do

  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl) 

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

     vlm(tl*tl+tl) = sum( llm_arr_p_lm0(:)*vtm(:,0) )
  end do

  spow_i(:) = 0.0

!$omp parallel do default(none) &
!$omp private(j, tl, tm, scal, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_m_lm0, llm_arr_m_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p, llm_arr_m) schedule(dynamic, 1) &
!$omp shared(htttht, costht, sintht, sfac, s, zl, ntht, tht, vlm, vtm)
  do tm=1,lmax
     do m=m+1,tm
        if (m<=s) then
           tfac = -sqrt( 1.0 * (s-m+1.) / (s+m) )
           llm_arr_p(:) = +llm_arr_p(:) * htttht(:) * tfac
           llm_arr_m(:) = +llm_arr_m(:) / htttht(:) * tfac
           l = s
        else
           tfac = +sqrt( 1.0 * m * (2.*m+1.)/(2.*(m+s)*(m-s)) )
           llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
           llm_arr_m(:) = +llm_arr_m(:) * tfac * sintht(:)
           l = m

           do j=1,ntht
              if (abs(llm_arr_p(j)) < 1./sfac) then
                 llm_arr_p(j) = llm_arr_p(j)*sfac
                 llm_arr_m(j) = llm_arr_m(j)*sfac
                 spow_i(j) = spow_i(j)-1
              end if
           end do
        end if
     end do
     m = tm

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.0

     llm_arr_m_lm0(:) = llm_arr_m(:)
     llm_arr_m_lm1(:) = 0.0

     vlm(l*l+l+m) = sum( llm_arr_p_lm0(:)*scal(:)*vtm(:,+m) )
     vlm(l*l+l-m) = sum( llm_arr_m_lm0(:)*scal(:)*vtm(:,-m) )

     rl(:) = 0.0
     do tl=l+1,lmax
        rl(tl) = sqrt( 1. * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.)) )
     end do

     do tl=l+1,lmax
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * (costht(:) + m*zl(tl)) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

        llm_arr_x_lmt(:) = (llm_arr_m_lm0(:) * (costht(:) - m*zl(tl)) - llm_arr_m_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_m_lm1(:) = llm_arr_m_lm0(:)
        llm_arr_m_lm0(:) = llm_arr_x_lmt(:)

        vlm(tl*tl+tl+m) = sum( llm_arr_p_lm0(:)*scal(:)*vtm(:,+m) )
        vlm(tl*tl+tl-m) = sum( llm_arr_m_lm0(:)*scal(:)*vtm(:,-m) )

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 llm_arr_m_lm0(j) = llm_arr_m_lm0(j)/sfac
                 llm_arr_m_lm1(j) = llm_arr_m_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine vtm2vlm
