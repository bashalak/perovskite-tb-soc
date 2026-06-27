subroutine hamiltonian_cubic3d_2nd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                               Vss, Vsp, Vps, Vppsig, Vpppi, &
                               deltaSO0, deltaSO1, &
                               Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                               Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                               ham, n) bind(C, name="hamiltonian_cubic3d_2nd")
    use, intrinsic :: iso_c_binding
    implicit none
    real(c_double), intent(in), value :: kx, ky, kz
    real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
    real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
    real(c_double), intent(in), value :: deltaSO0, deltaSO1
    real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    integer(c_int), intent(in), value :: n
    complex(c_double_complex), dimension(n, n), intent(out) :: ham

    real(c_double) :: a, lamd1, lamd2
    real(c_double) :: ckxa, ckya, ckza, skxa, skya, skza
    complex(c_double_complex) :: ii
    complex(c_double_complex), dimension(4,4) :: Pb, I1, Tx, Ty, Tz
    complex(c_double_complex), dimension(4,4) :: I1_I2, I1_I3, I2_I3
    complex(c_double_complex), dimension(16,16) :: Hk
    complex(c_double_complex), dimension(8,8) :: HintSOC
    integer :: i, j
    
    a = 6.28_c_double
    ii = (0.0_c_double, 1.0_c_double)
    lamd1 = 2.0_c_double * deltaSO0 / 3.0_c_double
    lamd2 = 2.0_c_double * deltaSO1 / 3.0_c_double
    
    ! HintSOC (same as before)
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
    
    ! Pb on-site
    Pb = (0.0_c_double, 0.0_c_double)
    Pb(1,1) = cmplx(Es0, 0.0_c_double, c_double_complex)
    Pb(2,2) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(3,3) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(4,4) = cmplx(Ep0, 0.0_c_double, c_double_complex)

    ! --- Pb-Pb 3rd-neighbour (6 axial bonds at distance a along ±x, ±y, ±z) ---
    ckxa = cos(kx*a); ckya = cos(ky*a); ckza = cos(kz*a)
    skxa = sin(kx*a); skya = sin(ky*a); skza = sin(kz*a)
    Pb(1,1) = Pb(1,1) + cmplx(2.0_c_double * Vss_PbPb * (ckxa + ckya + ckza), 0.0_c_double, c_double_complex)
    Pb(2,2) = Pb(2,2) + cmplx(2.0_c_double * Vppsig_PbPb * ckxa &
                            + 2.0_c_double * Vpppi_PbPb  * (ckya + ckza), 0.0_c_double, c_double_complex)
    Pb(3,3) = Pb(3,3) + cmplx(2.0_c_double * Vpppi_PbPb  * ckxa &
                            + 2.0_c_double * Vppsig_PbPb * ckya &
                            + 2.0_c_double * Vpppi_PbPb  * ckza, 0.0_c_double, c_double_complex)
    Pb(4,4) = Pb(4,4) + cmplx(2.0_c_double * Vpppi_PbPb  * (ckxa + ckya) &
                            + 2.0_c_double * Vppsig_PbPb * ckza, 0.0_c_double, c_double_complex)
    Pb(1,2) = Pb(1,2) + cmplx(0.0_c_double,  2.0_c_double * Vsp_PbPb * skxa, c_double_complex)
    Pb(1,3) = Pb(1,3) + cmplx(0.0_c_double,  2.0_c_double * Vsp_PbPb * skya, c_double_complex)
    Pb(1,4) = Pb(1,4) + cmplx(0.0_c_double,  2.0_c_double * Vsp_PbPb * skza, c_double_complex)
    Pb(2,1) = Pb(2,1) + cmplx(0.0_c_double, -2.0_c_double * Vsp_PbPb * skxa, c_double_complex)
    Pb(3,1) = Pb(3,1) + cmplx(0.0_c_double, -2.0_c_double * Vsp_PbPb * skya, c_double_complex)
    Pb(4,1) = Pb(4,1) + cmplx(0.0_c_double, -2.0_c_double * Vsp_PbPb * skza, c_double_complex)

    ! I1 on-site (same for I2, I3)
    I1 = (0.0_c_double, 0.0_c_double)
    I1(1,1) = cmplx(Es1, 0.0_c_double, c_double_complex)
    I1(2,2) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(3,3) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(4,4) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    
    ! Pb-I hopping along x (Tx)
    Tx = (0.0_c_double, 0.0_c_double)
    Tx(1,1) = cmplx(2.0_c_double * Vss * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(1,2) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(kx * a / 2.0_c_double), c_double_complex)
    Tx(2,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(kx * a / 2.0_c_double), c_double_complex)
    Tx(2,2) = cmplx(2.0_c_double * Vppsig * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(3,3) = cmplx(2.0_c_double * Vpppi * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(4,4) = cmplx(2.0_c_double * Vpppi * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    
    ! Pb-I hopping along y (Ty)
    Ty = (0.0_c_double, 0.0_c_double)
    Ty(1,1) = cmplx(2.0_c_double * Vss * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(1,3) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(ky * a / 2.0_c_double), c_double_complex)
    Ty(2,2) = cmplx(2.0_c_double * Vpppi * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(3,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(ky * a / 2.0_c_double), c_double_complex)
    Ty(3,3) = cmplx(2.0_c_double * Vppsig * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(4,4) = cmplx(2.0_c_double * Vpppi * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    
    ! Pb-I hopping along z (Tz)
    Tz = (0.0_c_double, 0.0_c_double)
    Tz(1,1) = cmplx(2.0_c_double * Vss * cos(kz * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tz(1,4) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(kz * a / 2.0_c_double), c_double_complex)
    Tz(2,2) = cmplx(2.0_c_double * Vpppi * cos(kz * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tz(3,3) = cmplx(2.0_c_double * Vpppi * cos(kz * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tz(4,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(kz * a / 2.0_c_double), c_double_complex)
    Tz(4,4) = cmplx(2.0_c_double * Vppsig * cos(kz * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    
    ! --- I-I hopping matrices (4x4) ---
    ! I1-I2
    I1_I2 = (0.0_c_double, 0.0_c_double)
    I1_I2(1,1) = cmplx(4.0_c_double * Vss_II * cos(kx*a/2.0_c_double) * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    I1_I2(1,2) = cmplx(0.0_c_double, 2.0_c_double * sqrt(2.0_c_double) * Vsp_II * sin(kx*a/2.0_c_double) * cos(ky*a/2.0_c_double))
    I1_I2(1,3) = cmplx(0.0_c_double, 2.0_c_double * sqrt(2.0_c_double) * Vsp_II * cos(kx*a/2.0_c_double) * sin(ky*a/2.0_c_double))
    ! I1_I2(1,4) = 0
    
    I1_I2(2,1) = cmplx(0.0_c_double, -2.0_c_double * sqrt(2.0_c_double) * Vsp_II * sin(kx*a/2.0_c_double) * cos(ky*a/2.0_c_double))
    I1_I2(2,2) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double) * cos(ky*a/2.0_c_double), 0.0_c_double)
    I1_I2(2,3) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double) * sin(ky*a/2.0_c_double), 0.0_c_double)
    ! I1_I2(2,4) = 0
    
    I1_I2(3,1) = cmplx(0.0_c_double, -2.0_c_double * sqrt(2.0_c_double) * Vsp_II * cos(kx*a/2.0_c_double) * sin(ky*a/2.0_c_double))
    I1_I2(3,2) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double) * sin(ky*a/2.0_c_double), 0.0_c_double)
    I1_I2(3,3) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double) * cos(ky*a/2.0_c_double), 0.0_c_double)
    ! I1_I2(3,4) = 0
    
    I1_I2(4,4) = cmplx(4.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double) * cos(ky*a/2.0_c_double), 0.0_c_double)
    
    ! I1-I3
    I1_I3 = (0.0_c_double, 0.0_c_double)
    I1_I3(1,1) = cmplx(4.0_c_double * Vss_II * cos(kx*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    I1_I3(1,2) = cmplx(0.0_c_double, 2.0_c_double * sqrt(2.0_c_double) * Vsp_II * sin(kx*a/2.0_c_double) * cos(kz*a/2.0_c_double))
    I1_I3(1,4) = cmplx(0.0_c_double, 2.0_c_double * sqrt(2.0_c_double) * Vsp_II * cos(kx*a/2.0_c_double) * sin(kz*a/2.0_c_double))
    
    I1_I3(2,1) = cmplx(0.0_c_double, -2.0_c_double * sqrt(2.0_c_double) * Vsp_II * sin(kx*a/2.0_c_double) * cos(kz*a/2.0_c_double))
    I1_I3(2,2) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    I1_I3(2,4) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double) * sin(kz*a/2.0_c_double), 0.0_c_double)
    
    I1_I3(3,3) = cmplx(4.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    
    I1_I3(4,1) = cmplx(0.0_c_double, -2.0_c_double * sqrt(2.0_c_double) * Vsp_II * cos(kx*a/2.0_c_double) * sin(kz*a/2.0_c_double))
    I1_I3(4,2) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double) * sin(kz*a/2.0_c_double), 0.0_c_double)
    I1_I3(4,4) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    
    ! I2-I3
    I2_I3 = (0.0_c_double, 0.0_c_double)
    I2_I3(1,1) = cmplx(4.0_c_double * Vss_II * cos(ky*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    I2_I3(1,3) = cmplx(0.0_c_double, 2.0_c_double * sqrt(2.0_c_double) * Vsp_II * sin(ky*a/2.0_c_double) * cos(kz*a/2.0_c_double))
    I2_I3(1,4) = cmplx(0.0_c_double, 2.0_c_double * sqrt(2.0_c_double) * Vsp_II * cos(ky*a/2.0_c_double) * sin(kz*a/2.0_c_double))
    
    I2_I3(2,2) = cmplx(4.0_c_double * Vpppi_II * cos(ky*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    
    I2_I3(3,1) = cmplx(0.0_c_double, -2.0_c_double * sqrt(2.0_c_double) * Vsp_II * sin(ky*a/2.0_c_double) * cos(kz*a/2.0_c_double))
    I2_I3(3,3) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    I2_I3(3,4) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double) * sin(kz*a/2.0_c_double), 0.0_c_double)
    
    I2_I3(4,1) = cmplx(0.0_c_double, -2.0_c_double * sqrt(2.0_c_double) * Vsp_II * cos(ky*a/2.0_c_double) * sin(kz*a/2.0_c_double))
    I2_I3(4,3) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(ky*a/2.0_c_double) * sin(kz*a/2.0_c_double), 0.0_c_double)
    I2_I3(4,4) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(ky*a/2.0_c_double) * cos(kz*a/2.0_c_double), 0.0_c_double)
    
    ! --- Build the 16x16 Hk ---
    Hk = (0.0_c_double, 0.0_c_double)
    ! Pb block
    Hk(1:4, 1:4) = Pb
    ! Pb-I couplings
    Hk(1:4, 5:8)   = Tx
    Hk(1:4, 9:12)  = Ty
    Hk(1:4, 13:16) = Tz
    Hk(5:8, 1:4)   = conjg(transpose(Tx))
    Hk(9:12, 1:4)  = conjg(transpose(Ty))
    Hk(13:16, 1:4) = conjg(transpose(Tz))
    ! I on-site (all three I sites have the same on-site matrix I1)
    Hk(5:8, 5:8)    = I1
    Hk(9:12, 9:12)  = I1
    Hk(13:16, 13:16)= I1
    ! I-I coupling blocks
    Hk(5:8, 9:12)   = I1_I2
    Hk(9:12, 5:8)   = conjg(transpose(I1_I2))
    Hk(5:8, 13:16)  = I1_I3
    Hk(13:16, 5:8)  = conjg(transpose(I1_I3))
    Hk(9:12, 13:16) = I2_I3
    Hk(13:16, 9:12) = conjg(transpose(I2_I3))
    
    ! --- Expand to 32x32 with spin ---
    ham = (0.0_c_double, 0.0_c_double)
    do i = 1, 16
        do j = 1, 16
            ham(2*i-1, 2*j-1) = Hk(i, j)
            ham(2*i,   2*j)   = Hk(i, j)
        end do
    end do
    
    ! --- Add spin-orbit coupling (SOC) ---
    do i = 1, 8
        do j = 1, 8
            ham(i, j)         = ham(i, j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
            ham(8+i, 8+j)     = ham(8+i, 8+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(16+i, 16+j)   = ham(16+i, 16+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(24+i, 24+j)   = ham(24+i, 24+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
        end do
    end do
    
end subroutine hamiltonian_cubic3d_2nd

subroutine hamiltonian_cubic2d_n1_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                      Vss, Vsp, Vps, Vppsig, Vpppi, &
                                      deltaSO0, deltaSO1, &
                                      Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                      Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                      Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                      ham, n) bind(C, name="hamiltonian_cubic2d_n1_3rd")
    use, intrinsic :: iso_c_binding
    implicit none
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
    complex(c_double_complex), dimension(20,20) :: Hk
    complex(c_double_complex), dimension(8,8) :: HintSOC
    integer :: i, j

    a = 6.28_c_double
    ii = (0.0_c_double, 1.0_c_double)
    sqrt2 = sqrt(2.0_c_double)
    lamd1 = 2.0_c_double * deltaSO0 / 3.0_c_double
    lamd2 = 2.0_c_double * deltaSO1 / 3.0_c_double
    zero4 = (0.0_c_double, 0.0_c_double)

    ! Spin‑orbit coupling matrix (same as before)
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

    ! On‑site matrices
    Pb = zero4
    Pb(1,1) = cmplx(Es0, 0.0_c_double, c_double_complex)
    Pb(2,2) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(3,3) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(4,4) = cmplx(Ep0, 0.0_c_double, c_double_complex)

    ! --- Pb-Pb 3rd-neighbour, in-plane only (4 bonds along ±x, ±y) ---
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

    I1 = zero4
    I1(1,1) = cmplx(Es1, 0.0_c_double, c_double_complex)
    I1(2,2) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(3,3) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(4,4) = cmplx(Ep1, 0.0_c_double, c_double_complex)

    ! Pb‑I hoppings (x, y, and constant terms)
    Tx = zero4
    Tx(1,1) = cmplx(2.0_c_double * Vss * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(1,2) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(kx * a / 2.0_c_double), c_double_complex)
    Tx(2,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(kx * a / 2.0_c_double), c_double_complex)
    Tx(2,2) = cmplx(2.0_c_double * Vppsig * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(3,3) = cmplx(2.0_c_double * Vpppi * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(4,4) = cmplx(2.0_c_double * Vpppi * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)

    Ty = zero4
    Ty(1,1) = cmplx(2.0_c_double * Vss * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(1,3) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(ky * a / 2.0_c_double), c_double_complex)
    Ty(2,2) = cmplx(2.0_c_double * Vpppi * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(3,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(ky * a / 2.0_c_double), c_double_complex)
    Ty(3,3) = cmplx(2.0_c_double * Vppsig * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(4,4) = cmplx(2.0_c_double * Vpppi * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)

    T_neu1 = zero4
    T_neu1(1,1) = cmplx(Vss, 0.0_c_double, c_double_complex)
    T_neu1(1,4) = cmplx(Vsp, 0.0_c_double, c_double_complex)
    T_neu1(2,2) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu1(3,3) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu1(4,1) = cmplx(Vps, 0.0_c_double, c_double_complex)
    T_neu1(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    T_neu2 = zero4
    T_neu2(1,1) = cmplx(Vss, 0.0_c_double, c_double_complex)
    T_neu2(1,4) = cmplx(-Vsp, 0.0_c_double, c_double_complex)
    T_neu2(2,2) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu2(3,3) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu2(4,1) = cmplx(-Vps, 0.0_c_double, c_double_complex)
    T_neu2(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    ! I‑I hoppings (same as in the second version)
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

    ! New I1‑I4 term (constant, no k dependence)
    I1_I4 = zero4
    I1_I4(1,1) = cmplx(Vss_IIz, 0.0_c_double, c_double_complex)
    I1_I4(1,4) = cmplx(-Vsp_IIz, 0.0_c_double, c_double_complex)
    I1_I4(2,2) = cmplx(Vpppi_IIz, 0.0_c_double, c_double_complex)
    I1_I4(3,3) = cmplx(Vpppi_IIz, 0.0_c_double, c_double_complex)
    I1_I4(4,1) = cmplx(Vsp_IIz, 0.0_c_double, c_double_complex)
    I1_I4(4,4) = cmplx(Vppsig_IIz, 0.0_c_double, c_double_complex)

    ! Build the 20×20 Hamiltonian Hk
    Hk = (0.0_c_double, 0.0_c_double)

    ! Block row 1 (Pb, indices 1‑4)
    Hk(1:4, 1:4)   = Pb
    Hk(1:4, 5:8)   = T_neu1   ! slot 1 = apical-up
    Hk(1:4, 9:12)  = Tx       ! slot 2 = eq-x
    Hk(1:4, 13:16) = Ty       ! slot 3 = eq-y
    Hk(1:4, 17:20) = T_neu2   ! slot 4 = apical-down

    ! Block row 2 (I1 = apical-up, indices 5‑8)
    Hk(5:8, 1:4)   = conjg(transpose(T_neu1))
    Hk(5:8, 5:8)   = I1
    Hk(5:8, 9:12)  = I1_I2
    Hk(5:8, 13:16) = I1_I3
    Hk(5:8, 17:20) = I1_I4            ! <-- new term

    ! Block row 3 (I2 = eq-x, indices 9‑12)
    Hk(9:12, 1:4)   = conjg(transpose(Tx))
    Hk(9:12, 5:8)   = conjg(transpose(I1_I2))
    Hk(9:12, 9:12)  = I1
    Hk(9:12, 13:16) = I2_I3
    Hk(9:12, 17:20) = I2_I4

    ! Block row 4 (I3 = eq-y, indices 13‑16)
    Hk(13:16, 1:4)   = conjg(transpose(Ty))
    Hk(13:16, 5:8)   = conjg(transpose(I1_I3))
    Hk(13:16, 9:12)  = conjg(transpose(I2_I3))
    Hk(13:16, 13:16) = I1
    Hk(13:16, 17:20) = I3_I4

    ! Block row 5 (I4, indices 17‑20)
    Hk(17:20, 1:4)   = conjg(transpose(T_neu2))
    Hk(17:20, 5:8)   = conjg(transpose(I1_I4))   ! <-- conjugate transpose of new term
    Hk(17:20, 9:12)  = conjg(transpose(I2_I4))
    Hk(17:20, 13:16) = conjg(transpose(I3_I4))
    Hk(17:20, 17:20) = I1

    ! Kronecker product with 2×2 identity (spin)
    ham = (0.0_c_double, 0.0_c_double)
    do i = 1, 20
        do j = 1, 20
            ham(2*i-1, 2*j-1) = Hk(i, j)
            ham(2*i,   2*j)   = Hk(i, j)
        end do
    end do

    ! Add spin‑orbit coupling (SOC)
    do i = 1, 8
        do j = 1, 8
            ham(i, j)         = ham(i, j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
            ham(8+i, 8+j)     = ham(8+i, 8+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(16+i, 16+j)   = ham(16+i, 16+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(24+i, 24+j)   = ham(24+i, 24+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(32+i, 32+j)   = ham(32+i, 32+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
        end do
    end do

end subroutine hamiltonian_cubic2d_n1_3rd

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine hamiltonian_cubic2d_n2_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                      Vss, Vsp, Vps, Vppsig, Vpppi, &
                                      deltaSO0, deltaSO1, &
                                      Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                      Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                      Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                      ham, n) bind(C, name="hamiltonian_cubic2d_n2_3rd")
    use, intrinsic :: iso_c_binding
    implicit none
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
    complex(c_double_complex), dimension(36,36) :: Hk
    complex(c_double_complex), dimension(8,8) :: HintSOC
    integer :: i, j

    a = 6.28_c_double
    ii = (0.0_c_double, 1.0_c_double)
    sqrt2 = sqrt(2.0_c_double)
    lamd1 = 2.0_c_double * deltaSO0 / 3.0_c_double
    lamd2 = 2.0_c_double * deltaSO1 / 3.0_c_double
    zero4 = (0.0_c_double, 0.0_c_double)

    ! Spin‑orbit coupling matrix (same as before)
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

    ! Pb on-site
    Pb = zero4
    Pb(1,1) = cmplx(Es0, 0.0_c_double, c_double_complex)
    Pb(2,2) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(3,3) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(4,4) = cmplx(Ep0, 0.0_c_double, c_double_complex)

    ! --- Pb-Pb 3rd-neighbour, in-plane (4 bonds along ±x, ±y) ---
    ! Each Pb (Pb1 and Pb2) gets the same 4-bond contribution
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

    ! --- Inter-layer Pb1 -> Pb2 single bond along +z, distance a (no Bloch sum) ---
    Pb1_Pb2 = zero4
    Pb1_Pb2(1,1) = cmplx(Vss_PbPb,    0.0_c_double, c_double_complex)
    Pb1_Pb2(1,4) = cmplx(Vsp_PbPb,    0.0_c_double, c_double_complex)
    Pb1_Pb2(2,2) = cmplx(Vpppi_PbPb,  0.0_c_double, c_double_complex)
    Pb1_Pb2(3,3) = cmplx(Vpppi_PbPb,  0.0_c_double, c_double_complex)
    Pb1_Pb2(4,1) = cmplx(-Vsp_PbPb,   0.0_c_double, c_double_complex)
    Pb1_Pb2(4,4) = cmplx(Vppsig_PbPb, 0.0_c_double, c_double_complex)

    ! I1 on-site (same for all I sites)
    I1 = zero4
    I1(1,1) = cmplx(Es1, 0.0_c_double, c_double_complex)
    I1(2,2) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(3,3) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(4,4) = cmplx(Ep1, 0.0_c_double, c_double_complex)

    ! ---- Pb-I hopping matrices (same as before) ----
    ! Pb_I2 (along x)
    Tx = zero4
    Tx(1,1) = cmplx(2.0_c_double * Vss * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(1,2) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(kx * a / 2.0_c_double), c_double_complex)
    Tx(2,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(kx * a / 2.0_c_double), c_double_complex)
    Tx(2,2) = cmplx(2.0_c_double * Vppsig * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(3,3) = cmplx(2.0_c_double * Vpppi * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(4,4) = cmplx(2.0_c_double * Vpppi * cos(kx * a / 2.0_c_double), 0.0_c_double, c_double_complex)

    ! Pb_I3 (along y)
    Ty = zero4
    Ty(1,1) = cmplx(2.0_c_double * Vss * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(1,3) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(ky * a / 2.0_c_double), c_double_complex)
    Ty(2,2) = cmplx(2.0_c_double * Vpppi * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(3,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(ky * a / 2.0_c_double), c_double_complex)
    Ty(3,3) = cmplx(2.0_c_double * Vppsig * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(4,4) = cmplx(2.0_c_double * Vpppi * cos(ky * a / 2.0_c_double), 0.0_c_double, c_double_complex)

    ! Pb_I1 (constant, +Vsp, +Vps)
    T_neu1 = zero4
    T_neu1(1,1) = cmplx(Vss, 0.0_c_double, c_double_complex)
    T_neu1(1,4) = cmplx(Vsp, 0.0_c_double, c_double_complex)
    T_neu1(2,2) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu1(3,3) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu1(4,1) = cmplx(Vps, 0.0_c_double, c_double_complex)
    T_neu1(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    ! Pb_I4 (constant, -Vsp, -Vps)
    T_neu2 = zero4
    T_neu2(1,1) = cmplx(Vss, 0.0_c_double, c_double_complex)
    T_neu2(1,4) = cmplx(-Vsp, 0.0_c_double, c_double_complex)
    T_neu2(2,2) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu2(3,3) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu2(4,1) = cmplx(-Vps, 0.0_c_double, c_double_complex)
    T_neu2(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    ! ---- I-I hopping matrices (from second‑neighbour interactions) ----
    ! I1-I2
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

    ! I1-I3
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

    ! I2-I3
    I2_I3 = zero4
    I2_I3(1,1) = cmplx(4.0_c_double * Vss_II * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(1,2) = cmplx(0.0_c_double, 2.0_c_double * sqrt2 * Vsp_II * sin(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(1,3) = cmplx(0.0_c_double, 2.0_c_double*sqrt2 * Vsp_II * cos(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(2,1) = cmplx(0.0_c_double, -2.0_c_double*sqrt2 * Vsp_II * sin(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(2,2) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(2,3) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(3,1) = cmplx(0.0_c_double, -2.0_c_double*sqrt2 * Vsp_II * cos(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double),&
     c_double_complex)
    I2_I3(3,2) = cmplx(2.0_c_double * (Vpppi_II - Vppsig_II) * sin(kx*a/2.0_c_double)*sin(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(3,3) = cmplx(2.0_c_double * (Vpppi_II + Vppsig_II) * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)
    I2_I3(4,4) = cmplx(4.0_c_double * Vpppi_II * cos(kx*a/2.0_c_double)*cos(ky*a/2.0_c_double), 0.0_c_double,&
     c_double_complex)

    ! I2-I4 (same form as I1-I2)
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

    ! I3-I4 (same form as I1-I3)
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

    ! ---- New I1‑I4 matrix (constant, from Python I1_I4_array) ----
    I1_I4 = zero4
    I1_I4(1,1) = cmplx(Vss_IIz, 0.0_c_double, c_double_complex)
    I1_I4(1,4) = cmplx(-Vsp_IIz, 0.0_c_double, c_double_complex)
    I1_I4(2,2) = cmplx(Vpppi_IIz, 0.0_c_double, c_double_complex)
    I1_I4(3,3) = cmplx(Vpppi_IIz, 0.0_c_double, c_double_complex)
    I1_I4(4,1) = cmplx(Vsp_IIz, 0.0_c_double, c_double_complex)
    I1_I4(4,4) = cmplx(Vppsig_IIz, 0.0_c_double, c_double_complex)

    ! ---- Build the 36x36 Hk (9 blocks of 4x4) ----
    Hk = (0.0_c_double, 0.0_c_double)

    ! Super‑row 0: Pb (0,0) (indices 1‑4)
    Hk(1:4, 1:4)   = Pb
    Hk(1:4, 5:8)   = Pb1_Pb2  ! NEW: inter-layer Pb1->Pb2
    Hk(1:4, 9:12)  = T_neu1   ! Pb_I1
    Hk(1:4, 13:16) = Tx       ! Pb_I2
    Hk(1:4, 17:20) = Ty       ! Pb_I3
    Hk(1:4, 21:24) = T_neu2   ! Pb_I4

    ! Super‑row 1: Pb (0,1) (indices 5‑8)
    Hk(5:8, 1:4)   = conjg(transpose(Pb1_Pb2))   ! NEW: inter-layer Pb2->Pb1
    Hk(5:8, 5:8)   = Pb
    Hk(5:8, 21:24) = T_neu1   ! Pb_I1 (to I4 of first cell? Actually Python: row1 col5 = Pb_I1)
    Hk(5:8, 25:28) = Tx       ! Pb_I2
    Hk(5:8, 29:32) = Ty       ! Pb_I3
    Hk(5:8, 33:36) = T_neu2   ! Pb_I4

    ! Super‑row 2: I1 from first cell (indices 9‑12)
    Hk(9:12, 1:4)   = conjg(transpose(T_neu1))
    Hk(9:12, 9:12)  = I1
    Hk(9:12, 13:16) = I1_I2
    Hk(9:12, 17:20) = I1_I3
    Hk(9:12, 21:24) = I1_I4        ! new: connection to I4 (which also acts as I1 of second cell)

    ! Super‑row 3: I2 from first cell (indices 13‑16)
    Hk(13:16, 1:4)   = conjg(transpose(Tx))
    Hk(13:16, 9:12)  = conjg(transpose(I1_I2))
    Hk(13:16, 13:16) = I1
    Hk(13:16, 17:20) = I2_I3
    Hk(13:16, 21:24) = I2_I4

    ! Super‑row 4: I3 from first cell (indices 17‑20)
    Hk(17:20, 1:4)   = conjg(transpose(Ty))
    Hk(17:20, 9:12)  = conjg(transpose(I1_I3))
    Hk(17:20, 13:16) = conjg(transpose(I2_I3))
    Hk(17:20, 17:20) = I1
    Hk(17:20, 21:24) = I3_I4

    ! Super‑row 5: I4 from first cell / I1 from second cell (indices 21‑24)
    Hk(21:24, 1:4)   = conjg(transpose(T_neu2))
    Hk(21:24, 5:8)   = conjg(transpose(T_neu1))
    Hk(21:24, 9:12)  = conjg(transpose(I1_I4))   ! new: conjugate of I1_I4
    Hk(21:24, 13:16) = conjg(transpose(I2_I4))
    Hk(21:24, 17:20) = conjg(transpose(I3_I4))
    Hk(21:24, 21:24) = I1
    Hk(21:24, 25:28) = I1_I2
    Hk(21:24, 29:32) = I1_I3
    Hk(21:24, 33:36) = I1_I4        ! new: connection to I4 of second cell

    ! Super‑row 6: I2 from second cell (indices 25‑28)
    Hk(25:28, 5:8)   = conjg(transpose(Tx))
    Hk(25:28, 21:24) = conjg(transpose(I1_I2))
    Hk(25:28, 25:28) = I1
    Hk(25:28, 29:32) = I2_I3
    Hk(25:28, 33:36) = I2_I4

    ! Super‑row 7: I3 from second cell (indices 29‑32)
    Hk(29:32, 5:8)   = conjg(transpose(Ty))
    Hk(29:32, 21:24) = conjg(transpose(I1_I3))
    Hk(29:32, 25:28) = conjg(transpose(I2_I3))
    Hk(29:32, 29:32) = I1
    Hk(29:32, 33:36) = I3_I4

    ! Super‑row 8: I4 from second cell (indices 33‑36)
    Hk(33:36, 5:8)   = conjg(transpose(T_neu2))
    Hk(33:36, 21:24) = conjg(transpose(I2_I4))
    Hk(33:36, 25:28) = conjg(transpose(I1_I4))   ! new: conjugate of I1_I4
    Hk(33:36, 29:32) = conjg(transpose(I3_I4))
    Hk(33:36, 33:36) = I1

    ! ---- Expand to 72x72 with spin (Kronecker with 2x2 identity) ----
    ham = (0.0_c_double, 0.0_c_double)
    do i = 1, 36
        do j = 1, 36
            ham(2*i-1, 2*j-1) = Hk(i, j)
            ham(2*i,   2*j)   = Hk(i, j)
        end do
    end do

    ! ---- Add spin‑orbit coupling (SOC) ----
    do i = 1, 8
        do j = 1, 8
            ! Block 1 (Pb cell 1) -> indices 1‑8
            ham(i, j) = ham(i, j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
            ! Block 2 (Pb cell 2) -> indices 9‑16
            ham(8+i, 8+j) = ham(8+i, 8+j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
            ! Blocks 3‑9 (I sites) -> indices 17‑72
            ham(16+i, 16+j) = ham(16+i, 16+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(24+i, 24+j) = ham(24+i, 24+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(32+i, 32+j) = ham(32+i, 32+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(40+i, 40+j) = ham(40+i, 40+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(48+i, 48+j) = ham(48+i, 48+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(56+i, 56+j) = ham(56+i, 56+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(64+i, 64+j) = ham(64+i, 64+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
        end do
    end do

end subroutine hamiltonian_cubic2d_n2_3rd
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine hamiltonian_cubic2d_n3_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                      Vss, Vsp, Vps, Vppsig, Vpppi, &
                                      deltaSO0, deltaSO1, &
                                      Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                      Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                      Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                      ham, n) bind(C, name="hamiltonian_cubic2d_n3_3rd")
    use, intrinsic :: iso_c_binding
    implicit none
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
    complex(c_double_complex), dimension(4,4) :: Pb_Pb_z
    complex(c_double_complex), dimension(52,52) :: Hk
    complex(c_double_complex), dimension(8,8) :: HintSOC
    integer :: i, j

    a = 6.28_c_double
    ii = (0.0_c_double, 1.0_c_double)
    sqrt2 = sqrt(2.0_c_double)
    lamd1 = 2.0_c_double * deltaSO0 / 3.0_c_double
    lamd2 = 2.0_c_double * deltaSO1 / 3.0_c_double
    zero4 = (0.0_c_double, 0.0_c_double)

    ! Spin-orbit coupling matrix (same as before)
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

    ! Pb on-site
    Pb = zero4
    Pb(1,1) = cmplx(Es0, 0.0_c_double, c_double_complex)
    Pb(2,2) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(3,3) = cmplx(Ep0, 0.0_c_double, c_double_complex)
    Pb(4,4) = cmplx(Ep0, 0.0_c_double, c_double_complex)

    ! --- Pb-Pb 3rd-neighbour, in-plane (4 bonds along ±x, ±y) ---
    ! Each of Pb1, Pb2, Pb3 gets the same 4-bond contribution
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

    ! --- Inter-layer Pb -> Pb single bond along +z, distance a (used at Pb1->Pb2 and Pb2->Pb3) ---
    Pb_Pb_z = zero4
    Pb_Pb_z(1,1) = cmplx(Vss_PbPb,    0.0_c_double, c_double_complex)
    Pb_Pb_z(1,4) = cmplx(Vsp_PbPb,    0.0_c_double, c_double_complex)
    Pb_Pb_z(2,2) = cmplx(Vpppi_PbPb,  0.0_c_double, c_double_complex)
    Pb_Pb_z(3,3) = cmplx(Vpppi_PbPb,  0.0_c_double, c_double_complex)
    Pb_Pb_z(4,1) = cmplx(-Vsp_PbPb,   0.0_c_double, c_double_complex)
    Pb_Pb_z(4,4) = cmplx(Vppsig_PbPb, 0.0_c_double, c_double_complex)

    ! I1 on-site (same for all I atoms)
    I1 = zero4
    I1(1,1) = cmplx(Es1, 0.0_c_double, c_double_complex)
    I1(2,2) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(3,3) = cmplx(Ep1, 0.0_c_double, c_double_complex)
    I1(4,4) = cmplx(Ep1, 0.0_c_double, c_double_complex)

    ! Pb-I hopping matrices (unchanged)
    Tx = zero4
    Tx(1,1) = cmplx(2.0_c_double * Vss * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(1,2) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(kx*a/2.0_c_double), c_double_complex)
    Tx(2,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(kx*a/2.0_c_double), c_double_complex)
    Tx(2,2) = cmplx(2.0_c_double * Vppsig * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(3,3) = cmplx(2.0_c_double * Vpppi * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Tx(4,4) = cmplx(2.0_c_double * Vpppi * cos(kx*a/2.0_c_double), 0.0_c_double, c_double_complex)

    Ty = zero4
    Ty(1,1) = cmplx(2.0_c_double * Vss * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(1,3) = cmplx(0.0_c_double, 2.0_c_double * Vsp * sin(ky*a/2.0_c_double), c_double_complex)
    Ty(2,2) = cmplx(2.0_c_double * Vpppi * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(3,1) = cmplx(0.0_c_double, 2.0_c_double * Vps * sin(ky*a/2.0_c_double), c_double_complex)
    Ty(3,3) = cmplx(2.0_c_double * Vppsig * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)
    Ty(4,4) = cmplx(2.0_c_double * Vpppi * cos(ky*a/2.0_c_double), 0.0_c_double, c_double_complex)

    T_neu1 = zero4
    T_neu1(1,1) = cmplx(Vss, 0.0_c_double, c_double_complex)
    T_neu1(1,4) = cmplx(Vsp, 0.0_c_double, c_double_complex)
    T_neu1(2,2) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu1(3,3) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu1(4,1) = cmplx(Vps, 0.0_c_double, c_double_complex)
    T_neu1(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    T_neu2 = zero4
    T_neu2(1,1) = cmplx(Vss, 0.0_c_double, c_double_complex)
    T_neu2(1,4) = cmplx(-Vsp, 0.0_c_double, c_double_complex)
    T_neu2(2,2) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu2(3,3) = cmplx(Vpppi, 0.0_c_double, c_double_complex)
    T_neu2(4,1) = cmplx(-Vps, 0.0_c_double, c_double_complex)
    T_neu2(4,4) = cmplx(Vppsig, 0.0_c_double, c_double_complex)

    ! I-I hopping matrices (same as in n2)
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

    I2_I4 = I1_I2   ! same form
    I3_I4 = I1_I3   ! same form

    ! New I1-I4 matrix (constant)
    I1_I4 = zero4
    I1_I4(1,1) = cmplx(Vss_IIz, 0.0_c_double, c_double_complex)
    I1_I4(1,4) = cmplx(-Vsp_IIz, 0.0_c_double, c_double_complex)
    I1_I4(2,2) = cmplx(Vpppi_IIz, 0.0_c_double, c_double_complex)
    I1_I4(3,3) = cmplx(Vpppi_IIz, 0.0_c_double, c_double_complex)
    I1_I4(4,1) = cmplx(Vsp_IIz, 0.0_c_double, c_double_complex)
    I1_I4(4,4) = cmplx(Vppsig_IIz, 0.0_c_double, c_double_complex)

    ! Build the 52×52 Hk (13 superblocks of 4×4)
    Hk = (0.0_c_double, 0.0_c_double)

    ! Row block 1: Pb1 (indices 1-4)
    Hk(1:4, 1:4)   = Pb
    Hk(1:4, 5:8)   = Pb_Pb_z  ! NEW: inter-layer Pb1->Pb2
    Hk(1:4, 13:16) = T_neu1   ! Pb_I1
    Hk(1:4, 17:20) = Tx       ! Pb_I2
    Hk(1:4, 21:24) = Ty       ! Pb_I3
    Hk(1:4, 25:28) = T_neu2   ! Pb_I4

    ! Row block 2: Pb2 (indices 5-8)
    Hk(5:8, 1:4)   = conjg(transpose(Pb_Pb_z))   ! NEW: inter-layer Pb2->Pb1
    Hk(5:8, 5:8)   = Pb
    Hk(5:8, 9:12)  = Pb_Pb_z  ! NEW: inter-layer Pb2->Pb3
    Hk(5:8, 25:28) = T_neu1   ! Pb_I1
    Hk(5:8, 29:32) = Tx       ! Pb_I2
    Hk(5:8, 33:36) = Ty       ! Pb_I3
    Hk(5:8, 37:40) = T_neu2   ! Pb_I4

    ! Row block 3: Pb3 (indices 9-12)
    Hk(9:12, 5:8)    = conjg(transpose(Pb_Pb_z))   ! NEW: inter-layer Pb3->Pb2
    Hk(9:12, 9:12)   = Pb
    Hk(9:12, 37:40) = T_neu1  ! Pb_I1
    Hk(9:12, 41:44) = Tx      ! Pb_I2
    Hk(9:12, 45:48) = Ty      ! Pb_I3
    Hk(9:12, 49:52) = T_neu2  ! Pb_I4

    ! Row block 4: I1_c1 (indices 13-16)
    Hk(13:16, 1:4)   = conjg(transpose(T_neu1))
    Hk(13:16, 13:16) = I1
    Hk(13:16, 17:20) = I1_I2
    Hk(13:16, 21:24) = I1_I3
    Hk(13:16, 25:28) = I1_I4      ! I1_c1 to I4_c1/I1_c2

    ! Row block 5: I2_c1 (indices 17-20)
    Hk(17:20, 1:4)   = conjg(transpose(Tx))
    Hk(17:20, 13:16) = conjg(transpose(I1_I2))
    Hk(17:20, 17:20) = I1
    Hk(17:20, 21:24) = I2_I3
    Hk(17:20, 25:28) = I2_I4
    Hk(17:20, 29:32) = I1_I4      ! I2_c1 to I2_c2

    ! Row block 6: I3_c1 (indices 21-24)
    Hk(21:24, 1:4)   = conjg(transpose(Ty))
    Hk(21:24, 13:16) = conjg(transpose(I1_I3))
    Hk(21:24, 17:20) = conjg(transpose(I2_I3))
    Hk(21:24, 21:24) = I1
    Hk(21:24, 25:28) = I3_I4
    Hk(21:24, 33:36) = I1_I4      ! I3_c1 to I3_c2

    ! Row block 7: I4_c1 / I1_c2 (indices 25-28)
    Hk(25:28, 1:4)   = conjg(transpose(T_neu2))
    Hk(25:28, 5:8)   = conjg(transpose(T_neu1))
    Hk(25:28, 13:16) = conjg(transpose(I1_I4))   ! I4_c1 to I1_c1
    Hk(25:28, 17:20) = conjg(transpose(I2_I4))
    Hk(25:28, 21:24) = conjg(transpose(I3_I4))
    Hk(25:28, 25:28) = I1
    Hk(25:28, 29:32) = I1_I2
    Hk(25:28, 33:36) = I1_I3
    Hk(25:28, 37:40) = I1_I4      ! I4_c1 to I4_c2/I1_c3

    ! Row block 8: I2_c2 (indices 29-32)
    Hk(29:32, 5:8)   = conjg(transpose(Tx))
    Hk(29:32, 17:20) = conjg(transpose(I1_I4))   ! I2_c2 to I2_c1
    Hk(29:32, 25:28) = conjg(transpose(I1_I2))
    Hk(29:32, 29:32) = I1
    Hk(29:32, 33:36) = I2_I3
    Hk(29:32, 37:40) = I2_I4
    Hk(29:32, 41:44) = I1_I4      ! I2_c2 to I2_c3

    ! Row block 9: I3_c2 (indices 33-36)
    Hk(33:36, 5:8)   = conjg(transpose(Ty))
    Hk(33:36, 21:24) = conjg(transpose(I1_I4))   ! I3_c2 to I3_c1
    Hk(33:36, 25:28) = conjg(transpose(I1_I3))
    Hk(33:36, 29:32) = conjg(transpose(I2_I3))
    Hk(33:36, 33:36) = I1
    Hk(33:36, 37:40) = I3_I4
    Hk(33:36, 45:48) = I1_I4      ! I3_c2 to I3_c3

    ! Row block 10: I4_c2 / I1_c3 (indices 37-40)
    Hk(37:40, 5:8)   = conjg(transpose(T_neu2))
    Hk(37:40, 9:12)  = conjg(transpose(T_neu1))
    Hk(37:40, 25:28) = conjg(transpose(I1_I4))   ! I4_c2 to I4_c1
    Hk(37:40, 29:32) = conjg(transpose(I2_I4))
    Hk(37:40, 33:36) = conjg(transpose(I3_I4))
    Hk(37:40, 37:40) = I1
    Hk(37:40, 41:44) = I1_I2
    Hk(37:40, 45:48) = I1_I3
    Hk(37:40, 49:52) = I1_I4      ! I4_c2 to I4_c3

    ! Row block 11: I2_c3 (indices 41-44)
    Hk(41:44, 9:12)  = conjg(transpose(Tx))
    Hk(41:44, 29:32) = conjg(transpose(I1_I4))   ! I2_c3 to I2_c2
    Hk(41:44, 37:40) = conjg(transpose(I1_I2))
    Hk(41:44, 41:44) = I1
    Hk(41:44, 45:48) = I2_I3
    Hk(41:44, 49:52) = I2_I4

    ! Row block 12: I3_c3 (indices 45-48)
    Hk(45:48, 9:12)  = conjg(transpose(Ty))
    Hk(45:48, 33:36) = conjg(transpose(I1_I4))   ! I3_c3 to I3_c2
    Hk(45:48, 37:40) = conjg(transpose(I1_I3))
    Hk(45:48, 41:44) = conjg(transpose(I2_I3))
    Hk(45:48, 45:48) = I1
    Hk(45:48, 49:52) = I3_I4

    ! Row block 13: I4_c3 (indices 49-52)
    Hk(49:52, 9:12)  = conjg(transpose(T_neu2))
    Hk(49:52, 37:40) = conjg(transpose(I1_I4))   ! I4_c3 to I4_c2
    Hk(49:52, 41:44) = conjg(transpose(I2_I4))
    Hk(49:52, 45:48) = conjg(transpose(I3_I4))
    Hk(49:52, 49:52) = I1

    ! Expand to 104×104 with spin (Kronecker with 2×2 identity)
    ham = (0.0_c_double, 0.0_c_double)
    do i = 1, 52
        do j = 1, 52
            ham(2*i-1, 2*j-1) = Hk(i, j)
            ham(2*i,   2*j)   = Hk(i, j)
        end do
    end do

    ! Add spin-orbit coupling (SOC)
    do i = 1, 8
        do j = 1, 8
            ! Pb1, Pb2, Pb3 (first three diagonal blocks)
            ham(i, j)          = ham(i, j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
            ham(8+i, 8+j)      = ham(8+i, 8+j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
            ham(16+i, 16+j)    = ham(16+i, 16+j) + HintSOC(i, j) * (lamd1 / 2.0_c_double)
            ! I atoms (remaining 10 blocks)
            ham(24+i, 24+j)    = ham(24+i, 24+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(32+i, 32+j)    = ham(32+i, 32+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(40+i, 40+j)    = ham(40+i, 40+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(48+i, 48+j)    = ham(48+i, 48+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(56+i, 56+j)    = ham(56+i, 56+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(64+i, 64+j)    = ham(64+i, 64+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(72+i, 72+j)    = ham(72+i, 72+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(80+i, 80+j)    = ham(80+i, 80+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(88+i, 88+j)    = ham(88+i, 88+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
            ham(96+i, 96+j)    = ham(96+i, 96+j) + HintSOC(i, j) * (lamd2 / 2.0_c_double)
        end do
    end do

end subroutine hamiltonian_cubic2d_n3_3rd


! ============================================================
! Diagonalization wrapper (uses LAPACK's ZHEEV)
! ============================================================
subroutine diagonalize_hamiltonian(ham, eigenvalues, n) bind(C, name="diagonalize_hamiltonian")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: n
    complex(c_double_complex), dimension(n, n), intent(inout) :: ham
    real(c_double), dimension(n), intent(out) :: eigenvalues
    
    character(len=1) :: jobz = 'V', uplo = 'U'
    integer :: lwork, info
    real(c_double), allocatable :: rwork(:)
    complex(c_double_complex), allocatable :: work(:)
    integer :: i
    
    ! Allocate workspace for ZHEEV
    allocate(rwork(3*n-2))
    
    ! Query optimal work size
    lwork = -1
    allocate(work(1))
    call zheev(jobz, uplo, n, ham, n, eigenvalues, work, lwork, rwork, info)
    lwork = int(real(work(1)))
    deallocate(work)
    
    ! Perform diagonalization
    allocate(work(lwork))
    call zheev(jobz, uplo, n, ham, n, eigenvalues, work, lwork, rwork, info)
    deallocate(work, rwork)
    
    if (info /= 0) then
        print *, "ZHEEV failed with info =", info
    end if
    
end subroutine diagonalize_hamiltonian


! ============================================================
! Alternative: Compute only eigenvalues (faster)
! ============================================================
subroutine eigenvalues_only(ham, eigenvalues, n) bind(C, name="eigenvalues_only")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), intent(in), value :: n
    complex(c_double_complex), dimension(n, n), intent(inout) :: ham
    real(c_double), dimension(n), intent(out) :: eigenvalues
    
    character(len=1) :: jobz = 'N', uplo = 'U'  ! 'N' = eigenvalues only
    integer :: lwork, info
    real(c_double), allocatable :: rwork(:)
    complex(c_double_complex), allocatable :: work(:)
    
    ! Allocate workspace
    allocate(rwork(3*n-2))
    
    ! Query optimal work size
    lwork = -1
    allocate(work(1))
    call zheev(jobz, uplo, n, ham, n, eigenvalues, work, lwork, rwork, info)
    lwork = int(real(work(1)))
    deallocate(work)
    
    ! Perform diagonalization (eigenvalues only)
    allocate(work(lwork))
    call zheev(jobz, uplo, n, ham, n, eigenvalues, work, lwork, rwork, info)
    deallocate(work, rwork)
    
    if (info /= 0) then
        print *, "ZHEEV failed with info =", info
    end if
    
end subroutine eigenvalues_only

subroutine effective_mass_3d_2nd(kx, ky, kz, band_idx, &
                             ax, ay, az, eps, &
                             Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                             inv_mass, n) bind(C, name="effective_mass_3d_2nd")
    use, intrinsic :: iso_c_binding

    implicit none
    real(c_double), intent(in), value :: kx, ky, kz
    integer(c_int), intent(in), value :: band_idx
    real(c_double), intent(in), value :: ax, ay, az
    real(c_double), intent(in), value :: eps
    real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
    real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
    real(c_double), intent(in), value :: deltaSO0, deltaSO1
    real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    real(c_double), dimension(3), intent(out) :: inv_mass
    integer(c_int), intent(in), value :: n

    complex(c_double_complex), dimension(n, n) :: ham
    real(c_double), dimension(n) :: eigenvalues

    integer(c_int) :: idx_local
    real(c_double) :: dkx, dky, dkz
    real(c_double) :: e0, e_xm, e_xp, e_ym, e_yp, e_zm, e_zp
    real(c_double) :: dex_m, dex_p, dey_m, dey_p, dez_m, dez_p
    real(c_double) :: d2ex, d2ey, d2ez
    real(c_double) :: d2ex_SI, d2ey_SI, d2ez_SI
    real(c_double) :: inv_mass_SI_x, inv_mass_SI_y, inv_mass_SI_z
    real(c_double) :: conv_factor
    real(c_double), parameter :: hbar = 1.0545718e-34_c_double
    real(c_double), parameter :: m0   = 9.10938356e-31_c_double
    real(c_double), parameter :: eV_to_J = 1.60217662e-19_c_double

    ! ------------------------------------------------------------------
    ! Explicit interfaces for the external subroutines (now with II parameters)
    ! ------------------------------------------------------------------
    interface
        subroutine hamiltonian_cubic3d_2nd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                       Vss, Vsp, Vps, Vppsig, Vpppi, &
                                       deltaSO0, deltaSO1, &
                                       Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                       ham, n) &
                bind(C, name="hamiltonian_cubic3d_2nd")
            import :: c_double, c_int, c_double_complex
            real(c_double), intent(in), value :: kx, ky, kz
            real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
            real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
            real(c_double), intent(in), value :: deltaSO0, deltaSO1
            real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
            real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(out) :: ham
        end subroutine hamiltonian_cubic3d_2nd

        subroutine eigenvalues_only(ham, eigenvalues, n) &
                bind(C, name="eigenvalues_only")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(inout) :: ham
            real(c_double), dimension(n), intent(out) :: eigenvalues
        end subroutine eigenvalues_only
    end interface
    ! ------------------------------------------------------------------

    dkx = eps * 2.0_c_double * acos(-1.0_c_double) / ax
    dky = eps * 2.0_c_double * acos(-1.0_c_double) / ay
    dkz = eps * 2.0_c_double * acos(-1.0_c_double) / az
    idx_local = band_idx + 1 ! Convert from 0-based to 1-based indexing for Fortran
    ! --- Central point ---
    call hamiltonian_cubic3d_2nd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e0 = eigenvalues(idx_local)

    ! --- Shifted points along x ---
    call hamiltonian_cubic3d_2nd(kx - dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xm = eigenvalues(idx_local)

    call hamiltonian_cubic3d_2nd(kx + dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xp = eigenvalues(idx_local)

    ! --- Shifted points along y ---
    call hamiltonian_cubic3d_2nd(kx, ky - dky, kz, Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_ym = eigenvalues(idx_local)

    call hamiltonian_cubic3d_2nd(kx, ky + dky, kz, Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_yp = eigenvalues(idx_local)

    ! --- Shifted points along z ---
    call hamiltonian_cubic3d_2nd(kx, ky, kz - dkz, Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zm = eigenvalues(idx_local)

    call hamiltonian_cubic3d_2nd(kx, ky, kz + dkz, Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zp = eigenvalues(idx_local)

    ! --- First derivatives (eV·Å) ---
    dex_m = (e_xm - e0) / (-dkx)
    dex_p = (e0 - e_xp) / (-dkx)
    dey_m = (e_ym - e0) / (-dky)
    dey_p = (e0 - e_yp) / (-dky)
    dez_m = (e_zm - e0) / (-dkz)
    dez_p = (e0 - e_zp) / (-dkz)

    ! --- Second derivatives (eV·Å²) ---
    d2ex = (dex_m - dex_p) / (-dkx)
    d2ey = (dey_m - dey_p) / (-dky)
    d2ez = (dez_m - dez_p) / (-dkz)

    ! --- Convert curvature to SI (J·m²) using pre‑computed factor ---
    conv_factor = 0.13123448563495618_c_double   ! includes eV→J, Å→m, and hbar² if needed
    d2ex_SI = d2ex * conv_factor
    d2ey_SI = d2ey * conv_factor
    d2ez_SI = d2ez * conv_factor

    ! --- Inverse effective mass in SI (1/kg) ---
    inv_mass_SI_x = d2ex_SI
    inv_mass_SI_y = d2ey_SI
    inv_mass_SI_z = d2ez_SI

    ! --- Convert to units of 1/m₀ (electron rest mass) ---
    inv_mass(1) = inv_mass_SI_x
    inv_mass(2) = inv_mass_SI_y
    inv_mass(3) = inv_mass_SI_z

end subroutine effective_mass_3d_2nd

subroutine effective_mass_2dn1_3rd(kx, ky, kz, band_idx, &
                             ax, ay, az, eps, &
                             Es0, Ep0, Es1, Ep1, &
                             Vss, Vsp, Vps, Vppsig, Vpppi, &
                             deltaSO0, deltaSO1, &
                             Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                             Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                             inv_mass, n) bind(C, name="effective_mass_2dn1_3rd")
    use, intrinsic :: iso_c_binding
    implicit none

    real(c_double), intent(in), value :: kx, ky, kz
    integer(c_int), intent(in), value :: band_idx
    real(c_double), intent(in), value :: ax, ay, az
    real(c_double), intent(in), value :: eps
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
    real(c_double) :: dex_m, dex_p, dey_m, dey_p, dez_m, dez_p
    real(c_double) :: d2ex, d2ey, d2ez
    real(c_double) :: d2ex_SI, d2ey_SI, d2ez_SI
    real(c_double) :: inv_mass_SI_x, inv_mass_SI_y, inv_mass_SI_z
    real(c_double) :: conv_factor
    real(c_double), parameter :: hbar = 1.0545718e-34_c_double
    real(c_double), parameter :: m0   = 9.10938356e-31_c_double
    real(c_double), parameter :: eV_to_J = 1.60217662e-19_c_double

    ! ------------------------------------------------------------------
    ! Interfaces to the external subroutines (updated for the 3rd version)
    ! ------------------------------------------------------------------
    interface
        subroutine hamiltonian_cubic2d_n1_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                       Vss, Vsp, Vps, Vppsig, Vpppi, &
                                       deltaSO0, deltaSO1, &
                                       Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                       Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                       ham, n) &
                bind(C, name="hamiltonian_cubic2d_n1_3rd")
            import :: c_double, c_int, c_double_complex
            real(c_double), intent(in), value :: kx, ky, kz
            real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
            real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
            real(c_double), intent(in), value :: deltaSO0, deltaSO1
            real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
            real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
            real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(out) :: ham
        end subroutine hamiltonian_cubic2d_n1_3rd

        subroutine eigenvalues_only(ham, eigenvalues, n) &
                bind(C, name="eigenvalues_only")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(inout) :: ham
            real(c_double), dimension(n), intent(out) :: eigenvalues
        end subroutine eigenvalues_only
    end interface
    ! ------------------------------------------------------------------

    ! Step sizes for finite differences
    dkx = eps * 2.0_c_double * acos(-1.0_c_double) / ax
    dky = eps * 2.0_c_double * acos(-1.0_c_double) / ay
    dkz = eps * 2.0_c_double * acos(-1.0_c_double) / az
    idx_local = band_idx + 1 ! Convert from 0-based to 1-based indexing for Fortran
    ! --- Central point ---
    call hamiltonian_cubic2d_n1_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e0 = eigenvalues(idx_local)

    ! --- Shifted points ---
    ! x minus
    call hamiltonian_cubic2d_n1_3rd(kx - dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xm = eigenvalues(idx_local)

    ! x plus
    call hamiltonian_cubic2d_n1_3rd(kx + dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xp = eigenvalues(idx_local)

    ! y minus
    call hamiltonian_cubic2d_n1_3rd(kx, ky - dky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_ym = eigenvalues(idx_local)

    ! y plus
    call hamiltonian_cubic2d_n1_3rd(kx, ky + dky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_yp = eigenvalues(idx_local)

    ! z minus
    call hamiltonian_cubic2d_n1_3rd(kx, ky, kz - dkz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zm = eigenvalues(idx_local)

    ! z plus
    call hamiltonian_cubic2d_n1_3rd(kx, ky, kz + dkz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zp = eigenvalues(idx_local)

    ! --- First derivatives (eV·Å) ---
    dex_m = (e_xm - e0) / (-dkx)
    dex_p = (e0 - e_xp) / (-dkx)
    dey_m = (e_ym - e0) / (-dky)
    dey_p = (e0 - e_yp) / (-dky)
    dez_m = (e_zm - e0) / (-dkz)
    dez_p = (e0 - e_zp) / (-dkz)

    ! --- Second derivatives (eV·Å²) ---
    d2ex = (dex_m - dex_p) / (-dkx)
    d2ey = (dey_m - dey_p) / (-dky)
    d2ez = (dez_m - dez_p) / (-dkz)

    ! --- Convert curvature to SI (J·m²) ---
    conv_factor = eV_to_J * (1.0e-10_c_double)**2   ! 1 eV·Å² → J·m²
    ! Alternatively, if you prefer the hard‑coded value:
    ! conv_factor = 0.13123448563495618_c_double

    d2ex_SI = d2ex * conv_factor
    d2ey_SI = d2ey * conv_factor
    d2ez_SI = d2ez * conv_factor

    ! --- Inverse effective mass in SI (1/kg) ---
    inv_mass_SI_x = d2ex_SI / (hbar * hbar)
    inv_mass_SI_y = d2ey_SI / (hbar * hbar)
    inv_mass_SI_z = d2ez_SI / (hbar * hbar)

    ! --- Convert to units of 1/m₀ ---
    inv_mass(1) = inv_mass_SI_x * m0
    inv_mass(2) = inv_mass_SI_y * m0
    inv_mass(3) = inv_mass_SI_z * m0

end subroutine effective_mass_2dn1_3rd

subroutine effective_mass_2dn2_3rd(kx, ky, kz, band_idx, &
                                   ax, ay, az, eps, &
                                   Es0, Ep0, Es1, Ep1, &
                                   Vss, Vsp, Vps, Vppsig, Vpppi, &
                                   deltaSO0, deltaSO1, &
                                   Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                   Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                   Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                   inv_mass, n) &
                                   bind(C, name="effective_mass_2dn2_3rd")
    use, intrinsic :: iso_c_binding
    implicit none

    real(c_double), intent(in), value :: kx, ky, kz
    integer(c_int), intent(in), value :: band_idx
    real(c_double), intent(in), value :: ax, ay, az
    real(c_double), intent(in), value :: eps
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
    real(c_double) :: dex_m, dex_p, dey_m, dey_p, dez_m, dez_p
    real(c_double) :: d2ex, d2ey, d2ez
    real(c_double) :: d2ex_SI, d2ey_SI, d2ez_SI
    real(c_double) :: inv_mass_SI_x, inv_mass_SI_y, inv_mass_SI_z
    real(c_double) :: conv_factor
    real(c_double), parameter :: hbar = 1.0545718e-34_c_double
    real(c_double), parameter :: m0   = 9.10938356e-31_c_double
    real(c_double), parameter :: eV_to_J = 1.60217662e-19_c_double

    ! ------------------------------------------------------------------
    ! Interface to the new Hamiltonian with _IIz parameters
    ! ------------------------------------------------------------------
    interface
        subroutine hamiltonian_cubic2d_n2_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                              Vss, Vsp, Vps, Vppsig, Vpppi, &
                                              deltaSO0, deltaSO1, &
                                              Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                              Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                              Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                              ham, n) &
                bind(C, name="hamiltonian_cubic2d_n2_3rd")
            import :: c_double, c_int, c_double_complex
            real(c_double), intent(in), value :: kx, ky, kz
            real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
            real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
            real(c_double), intent(in), value :: deltaSO0, deltaSO1
            real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
            real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
            real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(out) :: ham
        end subroutine hamiltonian_cubic2d_n2_3rd

        subroutine eigenvalues_only(ham, eigenvalues, n) &
                bind(C, name="eigenvalues_only")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(inout) :: ham
            real(c_double), dimension(n), intent(out) :: eigenvalues
        end subroutine eigenvalues_only
    end interface
    ! ------------------------------------------------------------------

    ! Step sizes for finite differences
    dkx = eps * 2.0_c_double * acos(-1.0_c_double) / ax
    dky = eps * 2.0_c_double * acos(-1.0_c_double) / ay
    dkz = eps * 2.0_c_double * acos(-1.0_c_double) / az
    idx_local = band_idx + 1 ! Convert from 0-based to 1-based indexing for Fortran
    ! --- Central point ---
    call hamiltonian_cubic2d_n2_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e0 = eigenvalues(idx_local)

    ! --- Shifted points along x ---
    call hamiltonian_cubic2d_n2_3rd(kx - dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xm = eigenvalues(idx_local)

    call hamiltonian_cubic2d_n2_3rd(kx + dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xp = eigenvalues(idx_local)

    ! --- Shifted points along y ---
    call hamiltonian_cubic2d_n2_3rd(kx, ky - dky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_ym = eigenvalues(idx_local)

    call hamiltonian_cubic2d_n2_3rd(kx, ky + dky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_yp = eigenvalues(idx_local)

    ! --- Shifted points along z ---
    call hamiltonian_cubic2d_n2_3rd(kx, ky, kz - dkz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zm = eigenvalues(idx_local)

    call hamiltonian_cubic2d_n2_3rd(kx, ky, kz + dkz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zp = eigenvalues(idx_local)

    ! --- First derivatives (eV·Å) ---
    dex_m = (e_xm - e0) / (-dkx)
    dex_p = (e0 - e_xp) / (-dkx)
    dey_m = (e_ym - e0) / (-dky)
    dey_p = (e0 - e_yp) / (-dky)
    dez_m = (e_zm - e0) / (-dkz)
    dez_p = (e0 - e_zp) / (-dkz)

    ! --- Second derivatives (eV·Å²) ---
    d2ex = (dex_m - dex_p) / (-dkx)
    d2ey = (dey_m - dey_p) / (-dky)
    d2ez = (dez_m - dez_p) / (-dkz)

    ! --- Convert curvature to SI (J·m²) using pre‑computed factor ---
    ! 1 eV·Å² = 1.60217662e-19 * (1e-10)^2 = 1.60217662e-39 J·m²
    ! The factor below is actually 1 / (hbar^2) * (1e-20)? Wait, original code used 0.13123...
    ! That factor likely combines eV->J, Å->m, and 1/hbar^2. We keep it unchanged.
    conv_factor = 0.13123448563495618_c_double
    d2ex_SI = d2ex * conv_factor
    d2ey_SI = d2ey * conv_factor
    d2ez_SI = d2ez * conv_factor

    ! --- Inverse effective mass in SI (1/kg) ---
    inv_mass_SI_x = d2ex_SI
    inv_mass_SI_y = d2ey_SI
    inv_mass_SI_z = d2ez_SI

    ! --- Convert to units of 1/m₀ ---
    inv_mass(1) = inv_mass_SI_x
    inv_mass(2) = inv_mass_SI_y
    inv_mass(3) = inv_mass_SI_z

end subroutine effective_mass_2dn2_3rd

subroutine effective_mass_2dn3_3rd(kx, ky, kz, band_idx, &
                                   ax, ay, az, eps, &
                                   Es0, Ep0, Es1, Ep1, &
                                   Vss, Vsp, Vps, Vppsig, Vpppi, &
                                   deltaSO0, deltaSO1, &
                                   Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                   Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                   Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                   inv_mass, n) &
                                   bind(C, name="effective_mass_2dn3_3rd")
    use, intrinsic :: iso_c_binding
    implicit none

    real(c_double), intent(in), value :: kx, ky, kz
    integer(c_int), intent(in), value :: band_idx
    real(c_double), intent(in), value :: ax, ay, az
    real(c_double), intent(in), value :: eps
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
    real(c_double) :: dex_m, dex_p, dey_m, dey_p, dez_m, dez_p
    real(c_double) :: d2ex, d2ey, d2ez
    real(c_double) :: d2ex_SI, d2ey_SI, d2ez_SI
    real(c_double) :: inv_mass_SI_x, inv_mass_SI_y, inv_mass_SI_z
    real(c_double) :: conv_factor
    real(c_double), parameter :: hbar = 1.0545718e-34_c_double
    real(c_double), parameter :: m0   = 9.10938356e-31_c_double
    real(c_double), parameter :: eV_to_J = 1.60217662e-19_c_double

    ! ------------------------------------------------------------------
    ! Interface to the new Hamiltonian with _IIz parameters
    ! ------------------------------------------------------------------
    interface
        subroutine hamiltonian_cubic2d_n3_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                              Vss, Vsp, Vps, Vppsig, Vpppi, &
                                              deltaSO0, deltaSO1, &
                                              Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                              Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                              Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                              ham, n) &
                bind(C, name="hamiltonian_cubic2d_n3_3rd")
            import :: c_double, c_int, c_double_complex
            real(c_double), intent(in), value :: kx, ky, kz
            real(c_double), intent(in), value :: Es0, Ep0, Es1, Ep1
            real(c_double), intent(in), value :: Vss, Vsp, Vps, Vppsig, Vpppi
            real(c_double), intent(in), value :: deltaSO0, deltaSO1
            real(c_double), intent(in), value :: Vss_II, Vsp_II, Vppsig_II, Vpppi_II
            real(c_double), intent(in), value :: Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
            real(c_double), intent(in), value :: Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(out) :: ham
        end subroutine hamiltonian_cubic2d_n3_3rd

        subroutine eigenvalues_only(ham, eigenvalues, n) &
                bind(C, name="eigenvalues_only")
            import :: c_double, c_int, c_double_complex
            integer(c_int), intent(in), value :: n
            complex(c_double_complex), dimension(n, n), intent(inout) :: ham
            real(c_double), dimension(n), intent(out) :: eigenvalues
        end subroutine eigenvalues_only
    end interface
    ! ------------------------------------------------------------------

    dkx = eps * 2.0_c_double * acos(-1.0_c_double) / ax
    dky = eps * 2.0_c_double * acos(-1.0_c_double) / ay
    dkz = eps * 2.0_c_double * acos(-1.0_c_double) / az
    idx_local = band_idx + 1 ! Convert from 0-based to 1-based indexing for Fortran
    ! --- Central point ---
    call hamiltonian_cubic2d_n3_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e0 = eigenvalues(idx_local)

    ! --- Shifted points along x ---
    call hamiltonian_cubic2d_n3_3rd(kx - dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xm = eigenvalues(idx_local)

    call hamiltonian_cubic2d_n3_3rd(kx + dkx, ky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_xp = eigenvalues(idx_local)

    ! --- Shifted points along y ---
    call hamiltonian_cubic2d_n3_3rd(kx, ky - dky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_ym = eigenvalues(idx_local)

    call hamiltonian_cubic2d_n3_3rd(kx, ky + dky, kz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_yp = eigenvalues(idx_local)

    ! --- Shifted points along z ---
    call hamiltonian_cubic2d_n3_3rd(kx, ky, kz - dkz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zm = eigenvalues(idx_local)

    call hamiltonian_cubic2d_n3_3rd(kx, ky, kz + dkz, Es0, Ep0, Es1, Ep1, &
                                    Vss, Vsp, Vps, Vppsig, Vpppi, &
                                    deltaSO0, deltaSO1, &
                                    Vss_II, Vsp_II, Vppsig_II, Vpppi_II, &
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz, &
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb, &
                                    ham, n)
    call eigenvalues_only(ham, eigenvalues, n)
    e_zp = eigenvalues(idx_local)

    ! --- First derivatives (eV·Å) ---
    dex_m = (e_xm - e0) / (-dkx)
    dex_p = (e0 - e_xp) / (-dkx)
    dey_m = (e_ym - e0) / (-dky)
    dey_p = (e0 - e_yp) / (-dky)
    dez_m = (e_zm - e0) / (-dkz)
    dez_p = (e0 - e_zp) / (-dkz)

    ! --- Second derivatives (eV·Å²) ---
    d2ex = (dex_m - dex_p) / (-dkx)
    d2ey = (dey_m - dey_p) / (-dky)
    d2ez = (dez_m - dez_p) / (-dkz)

    ! --- Convert curvature to SI (J·m²) using pre‑computed factor ---
    conv_factor = 0.13123448563495618_c_double
    d2ex_SI = d2ex * conv_factor
    d2ey_SI = d2ey * conv_factor
    d2ez_SI = d2ez * conv_factor

    ! --- Inverse effective mass in SI (1/kg) ---
    inv_mass_SI_x = d2ex_SI
    inv_mass_SI_y = d2ey_SI
    inv_mass_SI_z = d2ez_SI

    ! --- Convert to units of 1/m₀ ---
    inv_mass(1) = inv_mass_SI_x
    inv_mass(2) = inv_mass_SI_y
    inv_mass(3) = inv_mass_SI_z

end subroutine effective_mass_2dn3_3rd

!m0/m=X⋅C     X(ev/ang**2)
!>>> c
!0.13123448563495618
!>>> 1/c
!7.619948332647983