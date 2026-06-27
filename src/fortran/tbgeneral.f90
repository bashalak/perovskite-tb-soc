!=======================================================================
!  Generic sp3 + SOC tight-binding Hamiltonian builder.
!
!  This routine performs ONE thing: given a precomputed neighbor list
!  (built in Python during setup) and packed parameter arrays, it builds
!  H(k) for arbitrary unit cells and arbitrary species.
!
!  Setup work (POSCAR parsing, periodicity detection, neighbor-list
!  construction, shell clustering) is done in Python -- see tbgeneral.py.
!
!  Conventions
!  -----------
!  * Per-atom basis: 4 orbitals  (s, px, py, pz)
!  * With spin (always on): each atomic block becomes 8x8.
!    Final H dim: n_ham = 8 * n_atoms.
!  * Hetero-pair Slater-Koster:
!      <A_s | H | B_p_a> = + d_a * Vsps_AB         (d_a = bond direction cosine A->B)
!      <A_p_a | H | B_s> = - d_a * Vsps_BA         (note: BA, not AB)
!    For homo-pair both Vsps_AB and Vsps_BA equal Vsps. For a hetero
!    bond stored with atom_i = species 'A' (alphabetically first),
!    Vsps_AB / Vsps_BA are taken straight from the parameter array.
!    If atom_i is species 'B' instead, the Python wrapper sets
!    bond_swap = 1 so we swap them on the fly.
!  * SOC:  on-site 8x8 atomic block scaled by lambda = 2*delta_so/3 / 2.
!  * Bloch phase:  exp(i k . R_cart) where R_cart is the lattice
!    translation in Cartesian.  Non-periodic directions are masked out
!    in Python (we never emit a bond with R != 0 along a non-periodic
!    axis), so no extra logic is needed here.
!=======================================================================

subroutine build_tb_hamiltonian( &
        n_atoms, n_species, species_idx, &
        es_per_species, ep_per_species, delta_so_per_species, &
        n_bonds, bond_atom_i, bond_atom_j, bond_dr_x, bond_dr_y, bond_dr_z, &
        bond_dist, bond_R_cart_x, bond_R_cart_y, bond_R_cart_z, &
        bond_param_idx, bond_swap, &
        n_pair_shells, V_sss, V_sps_ab, V_sps_ba, V_pps, V_ppp, &
        kx, ky, kz, &
        ham, n_ham) &
        bind(C, name="build_tb_hamiltonian")
    use, intrinsic :: iso_c_binding
    implicit none

    ! ---------- input ----------
    integer(c_int), intent(in), value :: n_atoms, n_species, n_bonds, n_pair_shells, n_ham
    integer(c_int), intent(in) :: species_idx(n_atoms)            ! 0-based
    real(c_double), intent(in) :: es_per_species(n_species)
    real(c_double), intent(in) :: ep_per_species(n_species)
    real(c_double), intent(in) :: delta_so_per_species(n_species)

    integer(c_int), intent(in) :: bond_atom_i(n_bonds), bond_atom_j(n_bonds)   ! 0-based
    real(c_double), intent(in) :: bond_dr_x(n_bonds), bond_dr_y(n_bonds), bond_dr_z(n_bonds)
    real(c_double), intent(in) :: bond_dist(n_bonds)
    real(c_double), intent(in) :: bond_R_cart_x(n_bonds), bond_R_cart_y(n_bonds), bond_R_cart_z(n_bonds)
    integer(c_int), intent(in) :: bond_param_idx(n_bonds)         ! 0-based, indexes into V_*
    integer(c_int), intent(in) :: bond_swap(n_bonds)              ! 0 or 1

    real(c_double), intent(in) :: V_sss(n_pair_shells), V_pps(n_pair_shells), V_ppp(n_pair_shells)
    real(c_double), intent(in) :: V_sps_ab(n_pair_shells), V_sps_ba(n_pair_shells)

    real(c_double), intent(in), value :: kx, ky, kz

    ! ---------- output ----------
    complex(c_double_complex), intent(out) :: ham(n_ham, n_ham)

    ! ---------- locals ----------
    integer :: ib, sp, sp_i, sp_j, p_idx, swap
    integer :: i_atom, j_atom, i_off, j_off, ii, jj
    real(c_double) :: dx, dy, dz, d, l, m, n_, ll, mm, nn
    real(c_double) :: vsss_, vsps_, vpsg_, vpps_, vppp_     ! local per-bond V values
    real(c_double) :: phase_arg
    complex(c_double_complex) :: phase, ii_unit
    complex(c_double_complex) :: B(4,4), Bdag(4,4)
    complex(c_double_complex) :: HintSOC(8,8)
    real(c_double) :: lamd
    real(c_double) :: pp_anis_x, pp_anis_y, pp_anis_z

    ham = (0.0_c_double, 0.0_c_double)
    ii_unit = (0.0_c_double, 1.0_c_double)

    ! ---------- 8x8 SOC template (atomic, basis = s up, px up, py up, pz up,
    !                                    s dn, px dn, py dn, pz dn) ----------
    HintSOC = (0.0_c_double, 0.0_c_double)
    HintSOC(3,5) = -ii_unit
    HintSOC(3,8) = (1.0_c_double, 0.0_c_double)
    HintSOC(4,6) =  ii_unit
    HintSOC(4,7) = (-1.0_c_double, 0.0_c_double)
    HintSOC(5,3) =  ii_unit
    HintSOC(5,8) = -ii_unit
    HintSOC(6,4) = -ii_unit
    HintSOC(6,7) = -ii_unit
    HintSOC(7,4) = (-1.0_c_double, 0.0_c_double)
    HintSOC(7,6) =  ii_unit
    HintSOC(8,3) = (1.0_c_double, 0.0_c_double)
    HintSOC(8,5) =  ii_unit

    ! ===================================================================
    !  Loop over bonds
    ! ===================================================================
    do ib = 1, n_bonds
        i_atom = bond_atom_i(ib) + 1     ! to 1-based
        j_atom = bond_atom_j(ib) + 1
        p_idx  = bond_param_idx(ib) + 1
        swap   = bond_swap(ib)

        dx = bond_dr_x(ib); dy = bond_dr_y(ib); dz = bond_dr_z(ib)
        d  = bond_dist(ib)

        ! Direction cosines from atom i to atom j
        if (d < 1.0e-10_c_double) then
            l = 0.0_c_double; m = 0.0_c_double; n_ = 0.0_c_double
        else
            l = dx / d; m = dy / d; n_ = dz / d
        end if
        ll = l*l; mm = m*m; nn = n_*n_

        ! Pull V values for this (pair, shell). If swap=1, swap AB <-> BA
        ! (because the dict stores ('A','B') with A < B alphabetically; if
        ! atom i has species B and atom j has species A, the bond direction
        ! is BA-style and we need the A->B / B->A integrals reversed).
        vsss_ = V_sss(p_idx)
        vpps_ = V_pps(p_idx)
        vppp_ = V_ppp(p_idx)
        if (swap == 0) then
            vsps_ = V_sps_ab(p_idx)        ! used for <i_s|H|j_p>
            vpsg_ = V_sps_ba(p_idx)        ! used for <i_p|H|j_s>
        else
            vsps_ = V_sps_ba(p_idx)
            vpsg_ = V_sps_ab(p_idx)
        end if

        ! ---------- 4x4 Slater-Koster matrix B(i_orb, j_orb) ----------
        ! Rows: orbitals on atom i  (s, px, py, pz)
        ! Cols: orbitals on atom j  (s, px, py, pz)
        B = (0.0_c_double, 0.0_c_double)
        ! s-s
        B(1,1) = cmplx(vsss_, 0.0_c_double, c_double_complex)
        ! s on i  ->  p on j  (positive direction-cosine prefactor)
        B(1,2) = cmplx( l  * vsps_, 0.0_c_double, c_double_complex)
        B(1,3) = cmplx( m  * vsps_, 0.0_c_double, c_double_complex)
        B(1,4) = cmplx( n_ * vsps_, 0.0_c_double, c_double_complex)
        ! p on i  ->  s on j  (NEGATIVE direction-cosine prefactor; uses BA integral)
        B(2,1) = cmplx(-l  * vpsg_, 0.0_c_double, c_double_complex)
        B(3,1) = cmplx(-m  * vpsg_, 0.0_c_double, c_double_complex)
        B(4,1) = cmplx(-n_ * vpsg_, 0.0_c_double, c_double_complex)
        ! p-p
        pp_anis_x = ll * vpps_ + (1.0_c_double - ll) * vppp_
        pp_anis_y = mm * vpps_ + (1.0_c_double - mm) * vppp_
        pp_anis_z = nn * vpps_ + (1.0_c_double - nn) * vppp_
        B(2,2) = cmplx(pp_anis_x,                    0.0_c_double, c_double_complex)
        B(3,3) = cmplx(pp_anis_y,                    0.0_c_double, c_double_complex)
        B(4,4) = cmplx(pp_anis_z,                    0.0_c_double, c_double_complex)
        B(2,3) = cmplx(l*m*(vpps_-vppp_),            0.0_c_double, c_double_complex)
        B(2,4) = cmplx(l*n_*(vpps_-vppp_),           0.0_c_double, c_double_complex)
        B(3,2) = B(2,3)
        B(3,4) = cmplx(m*n_*(vpps_-vppp_),           0.0_c_double, c_double_complex)
        B(4,2) = B(2,4)
        B(4,3) = B(3,4)

        ! Bloch phase exp(i k . R_cart)
        phase_arg = kx * bond_R_cart_x(ib) + ky * bond_R_cart_y(ib) + kz * bond_R_cart_z(ib)
        phase     = cmplx(cos(phase_arg), sin(phase_arg), c_double_complex)
        B = phase * B

        ! Hermitian conjugate (placed at the (j, i) block)
        do jj = 1, 4
            do ii = 1, 4
                Bdag(jj, ii) = conjg(B(ii, jj))
            end do
        end do

        ! ---------- accumulate (spinless layer first; spin doubled below) ----------
        ! Spinless basis position: orbitals 1..4 of atom k -> rows 4*(k-1)+1 .. 4*k
        ! We use ham temporarily in spinless 4N layout in slots that we will
        ! later spin-double. To keep code simple we use the FINAL 8N layout
        ! directly: spin-up takes rows 8(k-1)+1..8(k-1)+4, spin-dn takes
        ! rows 8(k-1)+5..8(k-1)+8.
        i_off = 8 * (i_atom - 1)
        j_off = 8 * (j_atom - 1)

        ! up-up block
        do jj = 1, 4
            do ii = 1, 4
                ham(i_off + ii, j_off + jj) = ham(i_off + ii, j_off + jj) + B(ii, jj)
                ham(j_off + jj, i_off + ii) = ham(j_off + jj, i_off + ii) + Bdag(jj, ii)
            end do
        end do
        ! dn-dn block (same hopping, opposite spin)
        do jj = 1, 4
            do ii = 1, 4
                ham(i_off + 4 + ii, j_off + 4 + jj) = ham(i_off + 4 + ii, j_off + 4 + jj) + B(ii, jj)
                ham(j_off + 4 + jj, i_off + 4 + ii) = ham(j_off + 4 + jj, i_off + 4 + ii) + Bdag(jj, ii)
            end do
        end do
    end do

    ! ===================================================================
    !  On-site energies  (per atom, both spins)
    ! ===================================================================
    do i_atom = 1, n_atoms
        sp = species_idx(i_atom) + 1            ! to 1-based
        i_off = 8 * (i_atom - 1)
        ! spin up
        ham(i_off + 1, i_off + 1) = ham(i_off + 1, i_off + 1) + cmplx(es_per_species(sp), 0.0_c_double, c_double_complex)
        ham(i_off + 2, i_off + 2) = ham(i_off + 2, i_off + 2) + cmplx(ep_per_species(sp), 0.0_c_double, c_double_complex)
        ham(i_off + 3, i_off + 3) = ham(i_off + 3, i_off + 3) + cmplx(ep_per_species(sp), 0.0_c_double, c_double_complex)
        ham(i_off + 4, i_off + 4) = ham(i_off + 4, i_off + 4) + cmplx(ep_per_species(sp), 0.0_c_double, c_double_complex)
        ! spin dn
        ham(i_off + 5, i_off + 5) = ham(i_off + 5, i_off + 5) + cmplx(es_per_species(sp), 0.0_c_double, c_double_complex)
        ham(i_off + 6, i_off + 6) = ham(i_off + 6, i_off + 6) + cmplx(ep_per_species(sp), 0.0_c_double, c_double_complex)
        ham(i_off + 7, i_off + 7) = ham(i_off + 7, i_off + 7) + cmplx(ep_per_species(sp), 0.0_c_double, c_double_complex)
        ham(i_off + 8, i_off + 8) = ham(i_off + 8, i_off + 8) + cmplx(ep_per_species(sp), 0.0_c_double, c_double_complex)
    end do

    ! ===================================================================
    !  SOC  (each atom's 8x8 atomic block += HintSOC * lambda/2)
    !
    !  Note: the HintSOC template is in the basis (s_up, px_up, py_up,
    !  pz_up, s_dn, px_dn, py_dn, pz_dn) which matches the 8-block layout
    !  we used above.
    ! ===================================================================
    do i_atom = 1, n_atoms
        sp = species_idx(i_atom) + 1
        lamd = 2.0_c_double * delta_so_per_species(sp) / 3.0_c_double
        i_off = 8 * (i_atom - 1)
        do jj = 1, 8
            do ii = 1, 8
                ham(i_off + ii, i_off + jj) = ham(i_off + ii, i_off + jj) + HintSOC(ii, jj) * (lamd / 2.0_c_double)
            end do
        end do
    end do

end subroutine build_tb_hamiltonian
