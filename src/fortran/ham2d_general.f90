!=======================================================================
!  Generalized 2D (Ruddlesden-Popper) tight-binding Hamiltonian for an
!  arbitrary number of perovskite layers n_layers.
!
!  Block layout (0-indexed atomic blocks of size 4):
!     0 .. nL-1       : Pb_1 .. Pb_{nL}
!     nL              : I1_c1     (topmost apical I)
!     nL + 1 + 3(c-1) : I2_cc     (equatorial-x of cell c)        c=1..nL
!     nL + 2 + 3(c-1) : I3_cc     (equatorial-y of cell c)
!     nL + 3 + 3(c-1) : I4_cc     (bottom apical of cell c)
!                                  = I1_c{c+1} (shared with next cell)
!
!  Total atomic blocks       : 4*nL + 1
!  Spinless matrix dimension : 16*nL + 4
!  With spin (final ham)     : 32*nL + 8
!
!  Two routines, both bind(C):
!    hamiltonian_cubic2d_general(...)
!    effective_mass_2d_general(...)
!
!  The 4x4 sub-blocks (Pb, I1, Tx, Ty, T_neu1, T_neu2, I1_I2, I1_I3,
!  I2_I3, I2_I4, I3_I4, I1_I4, Pb1_Pb2) are bit-for-bit identical to
!  the n=1,2,3 routines in ham2d_3rd.f90.
!=======================================================================

subroutine hamiltonian_cubic2d_general(n_layers, kx, ky, kz, &
                                       Es0, Ep0, Es1, Ep1, &
                                       Vss, Vsp, Vps, Vppsig, Vpppi, &
                                       deltaSO0, deltaSO1, &
                                       Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                       Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                       ham, n) &
        bind(C, name="hamiltonian_cubic2d_general")
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), intent(in), value :: n_layers
    real(c_double), intent(in), value :: kx, ky, kz
    real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
    real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
    real(c_double), intent(in), value :: deltaSO0, deltaSO1
    real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    integer(c_int), intent(in), value :: n
    complex(c_double_complex), dimension(n, n), intent(out) :: ham

    real(c_double) :: a, lamd1, lamd2, sqrt2
    real(c_double) :: ckxa, ckya, skxa, skya
    complex(c_double_complex) :: ii
    complex(c_double_complex), dimension(4,4) :: Pb, I1, zero4
    complex(c_double_complex), dimension(4,4) :: Tx, Ty, T_neu1, T_neu2
    complex(c_double_complex), dimension(4,4) :: I1_I2, I1_I3, I2_I3, I2_I4, I3_I4, I1_I4
    complex(c_double_complex), dimension(4,4) :: Pb1_Pb2
    complex(c_double_complex), dimension(8,8) :: HintSOC

    complex(c_double_complex), allocatable :: Hk(:,:)
    integer :: nL, M0, M, n_blocks
    integer :: c, i, j, b
    integer :: pbb, i1b, i2b, i3b, i4b, pbn
    integer :: r0

    a = 6.28_c_double
    ii = (0.0_c_double, 1.0_c_double)
    sqrt2 = sqrt(2.0_c_double)
    lamd1 = 2.0_c_double * deltaSO0 / 3.0_c_double
    lamd2 = 2.0_c_double * deltaSO1 / 3.0_c_double
    zero4 = (0.0_c_double, 0.0_c_double)

    nL       = n_layers
    n_blocks = 4*nL + 1            ! number of 4x4 atomic blocks
    M0       = 4 * n_blocks        ! spinless dimension = 16*nL + 4
    M        = 2 * M0              ! with spin           = 32*nL + 8

    ! Sanity: caller must pass n = 32*nL + 8
    if (n /= M) then
        ham = (0.0_c_double, 0.0_c_double)
        return
    end if

    allocate(Hk(M0, M0))
    Hk = (0.0_c_double, 0.0_c_double)

    ! --------------------- 8x8 SOC template (atomic) ---------------------
    HintSOC = (0.0_c_double, 0.0_c_double)
    HintSOC(3,5) = -ii
    HintSOC(3,8) = (1.0_c_double, 0.0_c_double)
    HintSOC(4,6) = ii
    HintSOC(4,7) = (-1.0_c_double, 0.0_c_double)
    HintSOC(5,3) = ii
    HintSOC(5,8) = -ii
    HintSOC(6,4) = -ii
    HintSOC(6,7) = -ii
    HintSOC(7,4) = (-1.0_c_double, 0.0_c_double)
    HintSOC(7,6) = ii
    HintSOC(8,3) = (1.0_c_double, 0.0_c_double)
    HintSOC(8,5) = ii

    ! --------------------- on-site Pb (with in-plane Pb-Pb) --------------
    Pb = zero4
    Pb(1,1) = cmplx(Es0, 0.0_c_double, c_double_complex)
    Pb(2,2) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(3,3) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(4,4) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    ckxa = cos(kx*a); ckya = cos(ky*a)
    skxa = sin(kx*a); skya = sin(ky*a)
    Pb(1,1) = Pb(1,1) + cmplx(2.0_c_double * Vss_PbPb * (ckxa + ckya), 0.0_c_double, c_double_complex)
    Pb(2,2) = Pb(2,2) + cmplx(2.0_c_double * Vppsig_PbPb * ckxa &
                            + 2.0_c_double * Vpppi_PbPb  * ckya, 0.0_c_double, c_double_complex)
    Pb(3,3) = Pb(3,3) + cmplx(2.0_c_double * Vpppi_PbPb  * ckxa &
                            + 2.0_c_double * Vppsig_PbPb * ckya, 0.0_c_double, c_double_complex)
    Pb(4,4) = Pb(4,4) + cmplx(2.0_c_double * Vpppi_PbPb  * (ckxa + ckya), 0.0_c_double, c_double_complex)
    Pb(1,2) = Pb(1,2) + cmplx(0.0_c_double,  2.0_c_double * Vsp_PbPb * skxa, c_double_complex)
    Pb(1,3) = Pb(1,3) + cmplx(0.0_c_double,  2.0_c_double * Vsp_PbPb * skya, c_double_complex)
    Pb(2,1) = Pb(2,1) + cmplx(0.0_c_double, -2.0_c_double * Vsp_PbPb * skxa, c_double_complex)
    Pb(3,1) = Pb(3,1) + cmplx(0.0_c_double, -2.0_c_double * Vsp_PbPb * skya, c_double_complex)

    ! --------------------- inter-layer Pb-Pb (single bond along +z, dist a) ----
    ! Constant matrix: same direction-cosine convention as in n=2/n=3 of ham2d_3rd.f90.
    Pb1_Pb2 = zero4
    Pb1_Pb2(1,1) = cmplx(Vss_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(1,4) = cmplx(Vsp_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(2,2) = cmplx(Vpppi_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(3,3) = cmplx(Vpppi_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(4,1) = cmplx(-Vsp_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(4,4) = cmplx(Vppsig_PbPb, 0.0_c_double, c_double_complex)

    ! --------------------- on-site I ------------------------------------
    I1 = zero4
    I1(1,1) = cmplx(Es1, 0.0_c_double, c_double_complex)
    I1(2,2) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(3,3) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(4,4) = cmplx(Ep1, 0.0_c_double, c_double_complex)

    ! --------------------- Pb-I (Tx, Ty, T_neu1, T_neu2) -----------------
    Tx = zero4
    Tx(1,1) = cmplx(2.0_c_double * Vss * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(1,2) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(kx*a/2.0_c_double), c_double_complex)
    Tx(2,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(kx*a/2.0_c_double), c_double_complex)
    Tx(2,2) = cmplx(2.0_c_double * Vppsig * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(3,3) = cmplx(2.0_c_double * Vpppi  * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(4,4) = cmplx(2.0_c_double * Vpppi  * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)

    Ty = zero4
    Ty(1,1) = cmplx(2.0_c_double * Vss * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(1,3) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(ky*a/2.0_c_double), c_double_complex)
    Ty(2,2) = cmplx(2.0_c_double * Vpppi  * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(3,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(ky*a/2.0_c_double), c_double_complex)
    Ty(3,3) = cmplx(2.0_c_double * Vppsig * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(4,4) = cmplx(2.0_c_double * Vpppi  * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)

    T_neu1 = zero4
    T_neu1(1,1) = cmplx(Vss,    0.0_c_double, c_double_complex)
    T_neu1(1,4) = cmplx(Vsp,    0.0_c_double, c_double_complex)
    T_neu1(2,2) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu1(3,3) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu1(4,1) = cmplx(Vps,    0.0_c_double, c_double_complex)
    T_neu1(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    T_neu2 = zero4
    T_neu2(1,1) = cmplx(Vss,    0.0_c_double, c_double_complex)
    T_neu2(1,4) = cmplx(-Vsp,   0.0_c_double, c_double_complex)
    T_neu2(2,2) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu2(3,3) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu2(4,1) = cmplx(-Vps,   0.0_c_double, c_double_complex)
    T_neu2(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    ! --------------------- I-I within-cell (II 2nd-neighbour) ------------
    I1_I2 = zero4
    I1_I2(1,1) = cmplx(2.0_c_double * Vss_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(1,2) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(1,4) = cmplx(-sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(2,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(2,2) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(2,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(3,3) = cmplx(2.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(4,1) = cmplx(sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(4,2) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)

    I1_I3 = zero4
    I1_I3(1,1) = cmplx(2.0_c_double * Vss_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(1,3) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(1,4) = cmplx(-sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(2,2) = cmplx(2.0_c_double * Vpppi_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(3,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(3,3) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(3,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(4,1) = cmplx(sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(4,3) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)

    I2_I3 = zero4
    I2_I3(1,1) = cmplx(4.0_c_double * Vss_II * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(1,2) = cmplx(0.0_c_double, 2.0_c_double*sqrt2 * Vsp_II * sin(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(1,3) = cmplx(0.0_c_double, 2.0_c_double*sqrt2 * Vsp_II * cos(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(2,1) = cmplx(0.0_c_double, -2.0_c_double*sqrt2 * Vsp_II * sin(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(2,2) = cmplx(2.0_c_double*(Vpppi_II+Vppsig_II)*cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(2,3) = cmplx(2.0_c_double*(Vpppi_II-Vppsig_II)*sin(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(3,1) = cmplx(0.0_c_double, -2.0_c_double*sqrt2 * Vsp_II * cos(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(3,2) = cmplx(2.0_c_double*(Vpppi_II-Vppsig_II)*sin(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(3,3) = cmplx(2.0_c_double*(Vpppi_II+Vppsig_II)*cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(4,4) = cmplx(4.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)

    I2_I4 = zero4
    I2_I4(1,1) = cmplx(2.0_c_double * Vss_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(1,2) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(1,4) = cmplx(-sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(2,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(2,2) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(2,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(3,3) = cmplx(2.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(4,1) = cmplx(sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(4,2) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)

    I3_I4 = zero4
    I3_I4(1,1) = cmplx(2.0_c_double * Vss_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(1,3) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(1,4) = cmplx(-sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(2,2) = cmplx(2.0_c_double * Vpppi_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(3,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(3,3) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(3,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(4,1) = cmplx(sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(4,3) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)

    ! --------------------- I-I across-z (constant) -----------------------
    I1_I4 = zero4
    I1_I4(1,1) = cmplx(Vss_IIz,    0.0_c_double, c_double_complex)
    I1_I4(1,4) = cmplx(-Vsp_IIz,   0.0_c_double, c_double_complex)
    I1_I4(2,2) = cmplx(Vpppi_IIz,  0.0_c_double, c_double_complex)
    I1_I4(3,3) = cmplx(Vpppi_IIz,  0.0_c_double, c_double_complex)
    I1_I4(4,1) = cmplx(Vsp_IIz,    0.0_c_double, c_double_complex)
    I1_I4(4,4) = cmplx(Vppsig_IIz, 0.0_c_double, c_double_complex)

    ! ============================== assemble Hk ==========================
    ! Block-index helpers (0-based atomic-block index):
    !     pb(c)  = c - 1                  for c=1..nL
    !     i1(c)  = nL                     if c == 1
    !            = nL + 3*(c-1)           if c >= 2  (= i4(c-1))
    !     i2(c)  = nL + 1 + 3*(c-1)
    !     i3(c)  = nL + 2 + 3*(c-1)
    !     i4(c)  = nL + 3 + 3*(c-1)
    do c = 1, nL
        pbb = c - 1
        if (c == 1) then
            i1b = nL
        else
            i1b = nL + 3*(c-1)        ! shared with previous cell's I4
        end if
        i2b = nL + 1 + 3*(c-1)
        i3b = nL + 2 + 3*(c-1)
        i4b = nL + 3 + 3*(c-1)

        ! Diagonal blocks
        Hk(4*pbb+1:4*pbb+4, 4*pbb+1:4*pbb+4) = Pb
        Hk(4*i1b+1:4*i1b+4, 4*i1b+1:4*i1b+4) = I1
        Hk(4*i2b+1:4*i2b+4, 4*i2b+1:4*i2b+4) = I1
        Hk(4*i3b+1:4*i3b+4, 4*i3b+1:4*i3b+4) = I1
        Hk(4*i4b+1:4*i4b+4, 4*i4b+1:4*i4b+4) = I1

        ! Pb_c -> I sites in cell c (and Hermitian conjugates)
        call put_block(Hk, M0, pbb, i1b, T_neu1)
        call put_block(Hk, M0, pbb, i2b, Tx)
        call put_block(Hk, M0, pbb, i3b, Ty)
        call put_block(Hk, M0, pbb, i4b, T_neu2)

        ! Within-cell I-I
        call put_block(Hk, M0, i1b, i2b, I1_I2)
        call put_block(Hk, M0, i1b, i3b, I1_I3)
        call put_block(Hk, M0, i1b, i4b, I1_I4)
        call put_block(Hk, M0, i2b, i3b, I2_I3)
        call put_block(Hk, M0, i2b, i4b, I2_I4)
        call put_block(Hk, M0, i3b, i4b, I3_I4)
    end do

    ! Inter-layer Pb-Pb and across-cell I-I (z-direction)
    do c = 1, nL - 1
        pbb  = c - 1                  ! Pb_c
        pbn  = c                      ! Pb_{c+1}
        i2b  = nL + 1 + 3*(c-1)       ! I2_cc
        i3b  = nL + 2 + 3*(c-1)       ! I3_cc
        ! I4_cc is shared with I1_c{c+1} so its couplings to atoms in cell
        ! c+1 are already produced by the within-cell-(c+1) iteration.
        ! We only need to add the equatorial I-I across z (I2 to I2, I3 to I3).
        call put_block(Hk, M0, pbb, pbn,           Pb1_Pb2)   ! Pb_c -> Pb_{c+1}
        call put_block(Hk, M0, i2b, nL + 1 + 3*c,  I1_I4)     ! I2_cc -> I2_c{c+1}
        call put_block(Hk, M0, i3b, nL + 2 + 3*c,  I1_I4)     ! I3_cc -> I3_c{c+1}
    end do

    ! ====================== spin doubling: ham = Hk ⊗ I_2 ===============
    ham = (0.0_c_double, 0.0_c_double)
    do i = 1, M0
        do j = 1, M0
            ham(2*i-1, 2*j-1) = Hk(i, j)
            ham(2*i,   2*j)   = Hk(i, j)
        end do
    end do

    ! ====================== add SOC on every atomic 8x8 diag block =======
    ! Pb blocks (b = 0..nL-1)  -> lamd1
    ! I  blocks (b = nL..n_blocks-1) -> lamd2
    do b = 0, n_blocks - 1
        r0 = 8 * b
        if (b < nL) then
            do i = 1, 8
                do j = 1, 8
                    ham(r0+i, r0+j) = ham(r0+i, r0+j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
                end do
            end do
        else
            do i = 1, 8
                do j = 1, 8
                    ham(r0+i, r0+j) = ham(r0+i, r0+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
                end do
            end do
        end if
    end do

    deallocate(Hk)

contains

    ! Place a 4x4 block at atomic-block coords (b1, b2) and its Hermitian
    ! conjugate at (b2, b1). b1, b2 are 0-based atomic-block indices.
    subroutine put_block(Hmat, ldM, b1, b2, B)
        integer, intent(in) :: ldM, b1, b2
        complex(c_double_complex), intent(inout) :: Hmat(ldM, ldM)
        complex(c_double_complex), intent(in)    :: B(4,4)
        integer :: ii_, jj_
        do ii_ = 1, 4
            do jj_ = 1, 4
                Hmat(4*b1 + ii_, 4*b2 + jj_) = B(ii_, jj_)
                Hmat(4*b2 + jj_, 4*b1 + ii_) = conjg(B(ii_, jj_))
            end do
        end do
    end subroutine put_block

end subroutine hamiltonian_cubic2d_general


!=======================================================================
!  2D Hamiltonian with surface-apical-I on-site correction.
!
!  Identical to hamiltonian_cubic2d_general EXCEPT the two surface
!  apical I atoms (top of slab = I1_c1, bottom of slab = I4_cnL) use
!  Es1_ap / Ep1_ap instead of Es1 / Ep1.  All hoppings are unchanged.
!
!  Physical motivation: the surface apical I faces the organic spacer
!  (low-epsilon medium), raising its on-site energy vs the interior
!  apical I which sits between two Pb layers.  Wannier extraction for
!  2D n=1 gives Es1_ap = -11.734 eV vs Es1_eq = -12.034 eV (+0.30 eV).
!
!  The correction scales as ~2/(n+1) -> 0 for n->inf, recovering the
!  3D bulk limit.  All Pb-I hoppings remain identical to the 3D set.
!
!  Extra parameters (appended after the 23 base params):
!    Es1_ap  : s on-site energy of the 2 surface apical I atoms
!    Ep1_ap  : p on-site energy of the 2 surface apical I atoms
!=======================================================================

subroutine hamiltonian_cubic2d_apical(n_layers, kx, ky, kz, &
                                      Es0, Ep0, Es1, Ep1, &
                                      Vss, Vsp, Vps, Vppsig, Vpppi, &
                                      deltaSO0, deltaSO1, &
                                      Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                      Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                      Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                      Es1_ap, Ep1_ap, &
                                      Ep0_surf, &
                                      ham, n) &
        bind(C, name="hamiltonian_cubic2d_apical")
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), intent(in), value :: n_layers
    real(c_double), intent(in), value :: kx, ky, kz
    real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
    real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
    real(c_double), intent(in), value :: deltaSO0, deltaSO1
    real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    real(c_double), intent(in), value :: Es1_ap, Ep1_ap
    real(c_double), intent(in), value :: Ep0_surf
    integer(c_int), intent(in), value :: n
    complex(c_double_complex), dimension(n, n), intent(out) :: ham

    real(c_double) :: a, lamd1, lamd2, sqrt2
    real(c_double) :: ckxa, ckya, skxa, skya
    complex(c_double_complex) :: ii
    complex(c_double_complex), dimension(4,4) :: Pb, Pb_surf, I1, I1_surf, zero4
    complex(c_double_complex), dimension(4,4) :: Tx, Ty, T_neu1, T_neu2
    complex(c_double_complex), dimension(4,4) :: I1_I2, I1_I3, I2_I3, I2_I4, I3_I4, I1_I4
    complex(c_double_complex), dimension(4,4) :: Pb1_Pb2
    complex(c_double_complex), dimension(8,8) :: HintSOC

    complex(c_double_complex), allocatable :: Hk(:,:)
    integer :: nL, M0, M, n_blocks
    integer :: c, i, j, b
    integer :: pbb, i1b, i2b, i3b, i4b, pbn
    integer :: r0

    a = 6.28_c_double
    ii = (0.0_c_double, 1.0_c_double)
    sqrt2 = sqrt(2.0_c_double)
    lamd1 = 2.0_c_double * deltaSO0 / 3.0_c_double
    lamd2 = 2.0_c_double * deltaSO1 / 3.0_c_double
    zero4 = (0.0_c_double, 0.0_c_double)

    nL       = n_layers
    n_blocks = 4*nL + 1
    M0       = 4 * n_blocks
    M        = 2 * M0

    if (n /= M) then
        ham = (0.0_c_double, 0.0_c_double)
        return
    end if

    allocate(Hk(M0, M0))
    Hk = (0.0_c_double, 0.0_c_double)

    ! SOC template
    HintSOC = (0.0_c_double, 0.0_c_double)
    HintSOC(3,5) = -ii
    HintSOC(3,8) = (1.0_c_double, 0.0_c_double)
    HintSOC(4,6) = ii
    HintSOC(4,7) = (-1.0_c_double, 0.0_c_double)
    HintSOC(5,3) = ii
    HintSOC(5,8) = -ii
    HintSOC(6,4) = -ii
    HintSOC(6,7) = -ii
    HintSOC(7,4) = (-1.0_c_double, 0.0_c_double)
    HintSOC(7,6) = ii
    HintSOC(8,3) = (1.0_c_double, 0.0_c_double)
    HintSOC(8,5) = ii

    ! on-site Pb (with in-plane Pb-Pb)
    Pb = zero4
    Pb(1,1) = cmplx(Es0, 0.0_c_double, c_double_complex)
    Pb(2,2) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(3,3) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(4,4) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    ckxa = cos(kx*a); ckya = cos(ky*a)
    skxa = sin(kx*a); skya = sin(ky*a)
    Pb(1,1) = Pb(1,1) + cmplx(2.0_c_double * Vss_PbPb * (ckxa + ckya), 0.0_c_double, c_double_complex)
    Pb(2,2) = Pb(2,2) + cmplx(2.0_c_double * Vppsig_PbPb * ckxa &
                            + 2.0_c_double * Vpppi_PbPb  * ckya, 0.0_c_double, c_double_complex)
    Pb(3,3) = Pb(3,3) + cmplx(2.0_c_double * Vpppi_PbPb  * ckxa &
                            + 2.0_c_double * Vppsig_PbPb * ckya, 0.0_c_double, c_double_complex)
    Pb(4,4) = Pb(4,4) + cmplx(2.0_c_double * Vpppi_PbPb  * (ckxa + ckya), 0.0_c_double, c_double_complex)
    Pb(1,2) = Pb(1,2) + cmplx(0.0_c_double,  2.0_c_double * Vsp_PbPb * skxa, c_double_complex)
    Pb(1,3) = Pb(1,3) + cmplx(0.0_c_double,  2.0_c_double * Vsp_PbPb * skya, c_double_complex)
    Pb(2,1) = Pb(2,1) + cmplx(0.0_c_double, -2.0_c_double * Vsp_PbPb * skxa, c_double_complex)
    Pb(3,1) = Pb(3,1) + cmplx(0.0_c_double, -2.0_c_double * Vsp_PbPb * skya, c_double_complex)

    ! inter-layer Pb-Pb
    Pb1_Pb2 = zero4
    Pb1_Pb2(1,1) = cmplx(Vss_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(1,4) = cmplx(Vsp_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(2,2) = cmplx(Vpppi_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(3,3) = cmplx(Vpppi_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(4,1) = cmplx(-Vsp_PbPb, 0.0_c_double, c_double_complex)
    Pb1_Pb2(4,4) = cmplx(Vppsig_PbPb, 0.0_c_double, c_double_complex)

    ! on-site surface Pb (top and bottom layers: replace Ep0 with Ep0_surf on p diagonals)
    Pb_surf = Pb
    Pb_surf(2,2) = Pb_surf(2,2) + cmplx(Ep0_surf - Ep0, 0.0_c_double, c_double_complex)
    Pb_surf(3,3) = Pb_surf(3,3) + cmplx(Ep0_surf - Ep0, 0.0_c_double, c_double_complex)
    Pb_surf(4,4) = Pb_surf(4,4) + cmplx(Ep0_surf - Ep0, 0.0_c_double, c_double_complex)

    ! on-site bulk I (equatorial + interior apical)
    I1 = zero4
    I1(1,1) = cmplx(Es1, 0.0_c_double, c_double_complex)
    I1(2,2) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(3,3) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(4,4) = cmplx(Ep1, 0.0_c_double, c_double_complex)

    ! on-site surface apical I (top and bottom of slab only)
    I1_surf = zero4
    I1_surf(1,1) = cmplx(Es1_ap, 0.0_c_double, c_double_complex)
    I1_surf(2,2) = cmplx(Ep1_ap, 0.0_c_double, c_double_complex)
    I1_surf(3,3) = cmplx(Ep1_ap, 0.0_c_double, c_double_complex)
    I1_surf(4,4) = cmplx(Ep1_ap, 0.0_c_double, c_double_complex)

    ! Pb-I hoppings (same as 3D, unchanged for all layers including edge)
    Tx = zero4
    Tx(1,1) = cmplx(2.0_c_double * Vss * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(1,2) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(kx*a/2.0_c_double), c_double_complex)
    Tx(2,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(kx*a/2.0_c_double), c_double_complex)
    Tx(2,2) = cmplx(2.0_c_double * Vppsig * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(3,3) = cmplx(2.0_c_double * Vpppi  * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(4,4) = cmplx(2.0_c_double * Vpppi  * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)

    Ty = zero4
    Ty(1,1) = cmplx(2.0_c_double * Vss * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(1,3) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(ky*a/2.0_c_double), c_double_complex)
    Ty(2,2) = cmplx(2.0_c_double * Vpppi  * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(3,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(ky*a/2.0_c_double), c_double_complex)
    Ty(3,3) = cmplx(2.0_c_double * Vppsig * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(4,4) = cmplx(2.0_c_double * Vpppi  * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)

    T_neu1 = zero4
    T_neu1(1,1) = cmplx(Vss,    0.0_c_double, c_double_complex)
    T_neu1(1,4) = cmplx(Vsp,    0.0_c_double, c_double_complex)
    T_neu1(2,2) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu1(3,3) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu1(4,1) = cmplx(Vps,    0.0_c_double, c_double_complex)
    T_neu1(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    T_neu2 = zero4
    T_neu2(1,1) = cmplx(Vss,    0.0_c_double, c_double_complex)
    T_neu2(1,4) = cmplx(-Vsp,   0.0_c_double, c_double_complex)
    T_neu2(2,2) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu2(3,3) = cmplx(Vpppi,  0.0_c_double, c_double_complex)
    T_neu2(4,1) = cmplx(-Vps,   0.0_c_double, c_double_complex)
    T_neu2(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    ! I-I hoppings (identical to general routine)
    I1_I2 = zero4
    I1_I2(1,1) = cmplx(2.0_c_double * Vss_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(1,2) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(1,4) = cmplx(-sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(2,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(2,2) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(2,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(3,3) = cmplx(2.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(4,1) = cmplx(sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(4,2) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I1_I2(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)

    I1_I3 = zero4
    I1_I3(1,1) = cmplx(2.0_c_double * Vss_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(1,3) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(1,4) = cmplx(-sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(2,2) = cmplx(2.0_c_double * Vpppi_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(3,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(3,3) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(3,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(4,1) = cmplx(sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I3(4,3) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I1_I3(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)

    I2_I3 = zero4
    I2_I3(1,1) = cmplx(4.0_c_double * Vss_II * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(1,2) = cmplx(0.0_c_double, 2.0_c_double*sqrt2 * Vsp_II * sin(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(1,3) = cmplx(0.0_c_double, 2.0_c_double*sqrt2 * Vsp_II * cos(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(2,1) = cmplx(0.0_c_double, -2.0_c_double*sqrt2 * Vsp_II * sin(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(2,2) = cmplx(2.0_c_double*(Vpppi_II+Vppsig_II)*cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(2,3) = cmplx(2.0_c_double*(Vpppi_II-Vppsig_II)*sin(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(3,1) = cmplx(0.0_c_double, -2.0_c_double*sqrt2 * Vsp_II * cos(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(3,2) = cmplx(2.0_c_double*(Vpppi_II-Vppsig_II)*sin(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(3,3) = cmplx(2.0_c_double*(Vpppi_II+Vppsig_II)*cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(4,4) = cmplx(4.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)

    I2_I4 = zero4
    I2_I4(1,1) = cmplx(2.0_c_double * Vss_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(1,2) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(1,4) = cmplx(-sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(2,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(2,2) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(2,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(3,3) = cmplx(2.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(4,1) = cmplx(sqrt2 * Vsp_II * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I2_I4(4,2) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double), c_double_complex)
    I2_I4(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)

    I3_I4 = zero4
    I3_I4(1,1) = cmplx(2.0_c_double * Vss_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(1,3) = cmplx(0.0_c_double, sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(1,4) = cmplx(-sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(2,2) = cmplx(2.0_c_double * Vpppi_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(3,1) = cmplx(0.0_c_double, -sqrt2 * Vsp_II * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(3,3) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(3,4) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(4,1) = cmplx(sqrt2 * Vsp_II * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I3_I4(4,3) = cmplx(0.0_c_double, (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double), c_double_complex)
    I3_I4(4,4) = cmplx((Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)

    ! I-I z-direction (across layers within slab)
    I1_I4 = zero4
    I1_I4(1,1) = cmplx(Vss_IIz,    0.0_c_double, c_double_complex)
    I1_I4(1,4) = cmplx(-Vsp_IIz,   0.0_c_double, c_double_complex)
    I1_I4(2,2) = cmplx(Vpppi_IIz,  0.0_c_double, c_double_complex)
    I1_I4(3,3) = cmplx(Vpppi_IIz,  0.0_c_double, c_double_complex)
    I1_I4(4,1) = cmplx(Vsp_IIz,    0.0_c_double, c_double_complex)
    I1_I4(4,4) = cmplx(Vppsig_IIz, 0.0_c_double, c_double_complex)

    ! assemble Hk: same structure as general, but surface apical I get I1_surf
    do c = 1, nL
        pbb = c - 1
        if (c == 1) then
            i1b = nL
        else
            i1b = nL + 3*(c-1)
        end if
        i2b = nL + 1 + 3*(c-1)
        i3b = nL + 2 + 3*(c-1)
        i4b = nL + 3 + 3*(c-1)

        ! Pb on-site (surface layers get Ep0_surf on p-orbitals)
        if (c == 1 .or. c == nL) then
            Hk(4*pbb+1:4*pbb+4, 4*pbb+1:4*pbb+4) = Pb_surf
        else
            Hk(4*pbb+1:4*pbb+4, 4*pbb+1:4*pbb+4) = Pb
        end if

        ! Top apical I: surface only for c==1 (block nL)
        if (c == 1) then
            Hk(4*i1b+1:4*i1b+4, 4*i1b+1:4*i1b+4) = I1_surf
        else
            Hk(4*i1b+1:4*i1b+4, 4*i1b+1:4*i1b+4) = I1
        end if

        ! Equatorial I: always bulk
        Hk(4*i2b+1:4*i2b+4, 4*i2b+1:4*i2b+4) = I1
        Hk(4*i3b+1:4*i3b+4, 4*i3b+1:4*i3b+4) = I1

        ! Bottom apical I: surface only for c==nL (block 4*nL)
        if (c == nL) then
            Hk(4*i4b+1:4*i4b+4, 4*i4b+1:4*i4b+4) = I1_surf
        else
            Hk(4*i4b+1:4*i4b+4, 4*i4b+1:4*i4b+4) = I1
        end if

        ! Pb -> I hoppings (unchanged from general)
        call put_block(Hk, M0, pbb, i1b, T_neu1)
        call put_block(Hk, M0, pbb, i2b, Tx)
        call put_block(Hk, M0, pbb, i3b, Ty)
        call put_block(Hk, M0, pbb, i4b, T_neu2)

        ! I-I within cell
        call put_block(Hk, M0, i1b, i2b, I1_I2)
        call put_block(Hk, M0, i1b, i3b, I1_I3)
        call put_block(Hk, M0, i1b, i4b, I1_I4)
        call put_block(Hk, M0, i2b, i3b, I2_I3)
        call put_block(Hk, M0, i2b, i4b, I2_I4)
        call put_block(Hk, M0, i3b, i4b, I3_I4)
    end do

    ! Inter-layer Pb-Pb and equatorial I-I across z
    do c = 1, nL - 1
        pbb  = c - 1
        pbn  = c
        i2b  = nL + 1 + 3*(c-1)
        i3b  = nL + 2 + 3*(c-1)
        call put_block(Hk, M0, pbb, pbn,           Pb1_Pb2)
        call put_block(Hk, M0, i2b, nL + 1 + 3*c,  I1_I4)
        call put_block(Hk, M0, i3b, nL + 2 + 3*c,  I1_I4)
    end do

    ! spin doubling
    ham = (0.0_c_double, 0.0_c_double)
    do i = 1, M0
        do j = 1, M0
            ham(2*i-1, 2*j-1) = Hk(i, j)
            ham(2*i,   2*j)   = Hk(i, j)
        end do
    end do

    ! SOC on every atomic 8x8 block
    do b = 0, n_blocks - 1
        r0 = 8 * b
        if (b < nL) then
            do i = 1, 8
                do j = 1, 8
                    ham(r0+i, r0+j) = ham(r0+i, r0+j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
                end do
            end do
        else
            do i = 1, 8
                do j = 1, 8
                    ham(r0+i, r0+j) = ham(r0+i, r0+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
                end do
            end do
        end if
    end do

    deallocate(Hk)

contains

    subroutine put_block(Hmat, ldM, b1, b2, B)
        integer, intent(in) :: ldM, b1, b2
        complex(c_double_complex), intent(inout) :: Hmat(ldM, ldM)
        complex(c_double_complex), intent(in)    :: B(4,4)
        integer :: ii_, jj_
        do ii_ = 1, 4
            do jj_ = 1, 4
                Hmat(4*b1 + ii_, 4*b2 + jj_) = B(ii_, jj_)
                Hmat(4*b2 + jj_, 4*b1 + ii_) = conjg(B(ii_, jj_))
            end do
        end do
    end subroutine put_block

end subroutine hamiltonian_cubic2d_apical


!=======================================================================
!  Effective mass for the apical-corrected 2D Hamiltonian.
!  Same finite-difference scheme as effective_mass_2d_general.
!=======================================================================
subroutine effective_mass_2d_apical(n_layers, kx, ky, kz, band_idx, &
                                    ax, ay, az, eps, &
                                    Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    Es1_ap, Ep1_ap, &
                                    Ep0_surf, &
                                    inv_mass, n) &
        bind(C, name="effective_mass_2d_apical")
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), intent(in), value :: n_layers
    real(c_double), intent(in), value :: kx, ky, kz
    integer(c_int), intent(in), value :: band_idx
    real(c_double), intent(in), value :: ax, ay, az, eps
    real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
    real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
    real(c_double), intent(in), value :: deltaSO0, deltaSO1
    real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    real(c_double), intent(in), value :: Es1_ap, Ep1_ap
    real(c_double), intent(in), value :: Ep0_surf
    real(c_double), dimension(3), intent(out) :: inv_mass
    integer(c_int), intent(in), value :: n

    complex(c_double_complex), dimension(n, n) :: ham
    real(c_double), dimension(n) :: eigenvalues

    integer(c_int) :: idx_local
    real(c_double) :: dkx, dky, dkz
    real(c_double) :: e0, e_xm, e_xp, e_ym, e_yp, e_zm, e_zp
    real(c_double) :: d2ex, d2ey, d2ez
    real(c_double) :: conv_factor
    real(c_double), parameter :: hbar    = 1.0545718e-34_c_double
    real(c_double), parameter :: m0      = 9.10938356e-31_c_double
    real(c_double), parameter :: eV_to_J = 1.60217662e-19_c_double

    interface
        subroutine hamiltonian_cubic2d_apical(n_layers, kx, ky, kz, &
                                       Es0, Ep0, Es1, Ep1, &
                                       Vss, Vsp, Vps, Vppsig, Vpppi, &
                                       deltaSO0, deltaSO1, &
                                       Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                       Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                       Es1_ap, Ep1_ap, &
                                       Ep0_surf, &
                                       ham, n) &
                bind(C, name="hamiltonian_cubic2d_apical")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n_layers
            real(c_double), intent(in), value :: kx, ky, kz
            real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
            real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
            real(c_double), intent(in), value :: deltaSO0, deltaSO1
            real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
            real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
            real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
            real(c_double), intent(in), value :: Es1_ap, Ep1_ap
            real(c_double), intent(in), value :: Ep0_surf
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(out) :: ham
        end subroutine hamiltonian_cubic2d_apical

        subroutine eigenvalues_only(ham, eigenvalues, n) &
                bind(C, name="eigenvalues_only")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(inout) :: ham
            real(c_double), dimension(n), intent(out) :: eigenvalues
        end subroutine eigenvalues_only
    end interface

    dkx = eps * 2.0_c_double * acos(-1.0_c_double) / ax
    dky = eps * 2.0_c_double * acos(-1.0_c_double) / ay
    dkz = eps * 2.0_c_double * acos(-1.0_c_double) / az
    idx_local = band_idx + 1

    call hamiltonian_cubic2d_apical(n_layers, kx, ky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
        Es1_ap, Ep1_ap, Ep0_surf, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e0 = eigenvalues(idx_local)

    call hamiltonian_cubic2d_apical(n_layers, kx - dkx, ky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
        Es1_ap, Ep1_ap, Ep0_surf, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xm = eigenvalues(idx_local)

    call hamiltonian_cubic2d_apical(n_layers, kx + dkx, ky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
        Es1_ap, Ep1_ap, Ep0_surf, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xp = eigenvalues(idx_local)

    call hamiltonian_cubic2d_apical(n_layers, kx, ky - dky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
        Es1_ap, Ep1_ap, Ep0_surf, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_ym = eigenvalues(idx_local)

    call hamiltonian_cubic2d_apical(n_layers, kx, ky + dky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
        Es1_ap, Ep1_ap, Ep0_surf, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_yp = eigenvalues(idx_local)

    call hamiltonian_cubic2d_apical(n_layers, kx, ky, kz - dkz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
        Es1_ap, Ep1_ap, Ep0_surf, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zm = eigenvalues(idx_local)

    call hamiltonian_cubic2d_apical(n_layers, kx, ky, kz + dkz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
        Es1_ap, Ep1_ap, Ep0_surf, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zp = eigenvalues(idx_local)

    d2ex = (e_xp - 2.0_c_double*e0 + e_xm) / (dkx*dkx)
    d2ey = (e_yp - 2.0_c_double*e0 + e_ym) / (dky*dky)
    d2ez = (e_zp - 2.0_c_double*e0 + e_zm) / (dkz*dkz)

    conv_factor = eV_to_J * (1.0e-10_c_double)**2
    inv_mass(1) = d2ex * conv_factor / (hbar * hbar) * m0
    inv_mass(2) = d2ey * conv_factor / (hbar * hbar) * m0
    inv_mass(3) = d2ez * conv_factor / (hbar * hbar) * m0

end subroutine effective_mass_2d_apical


!=======================================================================
!  Generalized 2D effective mass.
!  Builds the Hamiltonian at 7 k-points (central + ±dkx, ±dky, ±dkz),
!  diagonalises each, takes a centred 2nd-difference for the curvature,
!  returns inverse mass in units of 1/m0 along x, y, z.
!=======================================================================
subroutine effective_mass_2d_general(n_layers, kx, ky, kz, band_idx, &
                                     ax, ay, az, eps, &
                                     Es0, Ep0, Es1, Ep1, &
                                     Vss, Vsp, Vps, Vppsig, Vpppi, &
                                     deltaSO0, deltaSO1, &
                                     Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                     Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                     Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                     inv_mass, n) &
        bind(C, name="effective_mass_2d_general")
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), intent(in), value :: n_layers
    real(c_double), intent(in), value :: kx, ky, kz
    integer(c_int), intent(in), value :: band_idx
    real(c_double), intent(in), value :: ax, ay, az, eps
    real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
    real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
    real(c_double), intent(in), value :: deltaSO0, deltaSO1
    real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    real(c_double), dimension(3), intent(out) :: inv_mass
    integer(c_int), intent(in), value :: n

    complex(c_double_complex), dimension(n, n) :: ham
    real(c_double), dimension(n) :: eigenvalues

    integer(c_int) :: idx_local
    real(c_double) :: dkx, dky, dkz
    real(c_double) :: e0, e_xm, e_xp, e_ym, e_yp, e_zm, e_zp
    real(c_double) :: d2ex, d2ey, d2ez
    real(c_double) :: conv_factor
    real(c_double), parameter :: hbar    = 1.0545718e-34_c_double
    real(c_double), parameter :: m0      = 9.10938356e-31_c_double
    real(c_double), parameter :: eV_to_J = 1.60217662e-19_c_double

    interface
        subroutine hamiltonian_cubic2d_general(n_layers, kx, ky, kz, &
                                       Es0, Ep0, Es1, Ep1, &
                                       Vss, Vsp, Vps, Vppsig, Vpppi, &
                                       deltaSO0, deltaSO1, &
                                       Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                       Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                       ham, n) &
                bind(C, name="hamiltonian_cubic2d_general")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n_layers
            real(c_double), intent(in), value :: kx, ky, kz
            real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
            real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
            real(c_double), intent(in), value :: deltaSO0, deltaSO1
            real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
            real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
            real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(out) :: ham
        end subroutine hamiltonian_cubic2d_general

        subroutine eigenvalues_only(ham, eigenvalues, n) &
                bind(C, name="eigenvalues_only")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(inout) :: ham
            real(c_double), dimension(n), intent(out) :: eigenvalues
        end subroutine eigenvalues_only
    end interface

    dkx = eps * 2.0_c_double * acos(-1.0_c_double) / ax
    dky = eps * 2.0_c_double * acos(-1.0_c_double) / ay
    dkz = eps * 2.0_c_double * acos(-1.0_c_double) / az
    idx_local = band_idx + 1     ! Python 0-based -> Fortran 1-based

    ! central
    call hamiltonian_cubic2d_general(n_layers, kx, ky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e0 = eigenvalues(idx_local)

    ! x-
    call hamiltonian_cubic2d_general(n_layers, kx - dkx, ky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xm = eigenvalues(idx_local)

    ! x+
    call hamiltonian_cubic2d_general(n_layers, kx + dkx, ky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xp = eigenvalues(idx_local)

    ! y-
    call hamiltonian_cubic2d_general(n_layers, kx, ky - dky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_ym = eigenvalues(idx_local)

    ! y+
    call hamiltonian_cubic2d_general(n_layers, kx, ky + dky, kz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_yp = eigenvalues(idx_local)

    ! z-
    call hamiltonian_cubic2d_general(n_layers, kx, ky, kz - dkz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zm = eigenvalues(idx_local)

    ! z+
    call hamiltonian_cubic2d_general(n_layers, kx, ky, kz + dkz, &
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi, &
        deltaSO0, deltaSO1, &
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zp = eigenvalues(idx_local)

    ! Centred 2nd differences (eV / Å^2)
    d2ex = (e_xp - 2.0_c_double*e0 + e_xm) / (dkx*dkx)
    d2ey = (e_yp - 2.0_c_double*e0 + e_ym) / (dky*dky)
    d2ez = (e_zp - 2.0_c_double*e0 + e_zm) / (dkz*dkz)

    ! Convert eV·Å^2 -> J·m^2  then  1/m_eff = d2E / hbar^2  (in 1/kg)
    ! Result in units of 1/m0:
    conv_factor = eV_to_J * (1.0e-10_c_double)**2
    inv_mass(1) = d2ex * conv_factor / (hbar * hbar) * m0
    inv_mass(2) = d2ey * conv_factor / (hbar * hbar) * m0
    inv_mass(3) = d2ez * conv_factor / (hbar * hbar) * m0

end subroutine effective_mass_2d_general
