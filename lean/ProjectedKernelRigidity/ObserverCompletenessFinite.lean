import ProjectedKernelRigidity.ObserverCompleteness
import Mathlib.LinearAlgebra.Dual.Lemmas
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Algebra.Module.Projective
import Mathlib.Tactic

/-! Rank and algebraic-dual criteria for observer completeness. Only the rank
criterion needs finite dimension; the algebraic-dual criterion holds over any
field, using extension of linear functionals. -/

namespace ProjectedKernelRigidity

open LinearMap Submodule Module

variable {F V Y Z T : Type*} [Field F]
variable [AddCommGroup V] [Module F V]
variable [AddCommGroup Y] [Module F Y]
variable [AddCommGroup Z] [Module F Z]
variable [AddCommGroup T] [Module F T]

theorem observer_complete_iff_dual_image
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z) :
    Submodule.map O E.ker ≤ R ↔
      Submodule.map O.dualMap R.dualAnnihilator ≤ E.dualMap.range := by
  rw [LinearMap.range_dualMap_eq_dualAnnihilator_ker]
  constructor
  · intro h
    rintro _ ⟨c, hc, rfl⟩
    rw [Submodule.mem_dualAnnihilator]
    intro v hv
    exact (Submodule.mem_dualAnnihilator _).mp hc (O v) (h ⟨v, hv, rfl⟩)
  · intro h
    rintro _ ⟨v, hv, rfl⟩
    apply (Subspace.forall_mem_dualAnnihilator_apply_eq_zero_iff R (O v)).mp
    intro c hc
    exact (Submodule.mem_dualAnnihilator _).mp (h ⟨c, hc, rfl⟩) v hv

/-- A simultaneous linear choice of all the dual lifts. -/
structure ObserverDualCertificate
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z) where
  lift : R.dualAnnihilator →ₗ[F] Module.Dual F Y
  identity : E.dualMap.comp lift = O.dualMap.comp R.dualAnnihilator.subtype

theorem observerDualCertificate_nonempty_iff
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z) :
    Nonempty (ObserverDualCertificate E O R) ↔ Submodule.map O E.ker ≤ R := by
  rw [observer_complete_iff_dual_image]
  constructor
  · rintro ⟨cert⟩ _ ⟨c, hc, rfl⟩
    exact ⟨cert.lift ⟨c, hc⟩, LinearMap.congr_fun cert.identity ⟨c, hc⟩⟩
  · intro h
    classical
    let : Module.Projective F E.dualMap.range :=
      Module.Projective.of_basis (Module.Basis.ofVectorSpace F E.dualMap.range)
    let B := O.dualMap.comp R.dualAnnihilator.subtype
    let Br : R.dualAnnihilator →ₗ[F] E.dualMap.range :=
      B.codRestrict E.dualMap.range (fun c => h ⟨c, c.property, rfl⟩)
    obtain ⟨sectionMap, hs⟩ := E.dualMap.rangeRestrict.exists_rightInverse_of_surjective
      E.dualMap.range_rangeRestrict
    refine ⟨⟨sectionMap.comp Br, ?_⟩⟩
    ext c v
    have heq := congrArg Subtype.val (LinearMap.congr_fun hs (Br c))
    exact LinearMap.congr_fun heq v

/-- The stacked rank increment is exactly the dimension of the observed
kernel, independently of the supplied candidate panel. -/
theorem observer_rank_increment [FiniteDimensional F V]
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) :
    finrank F (E.prod O).range - finrank F E.range =
      finrank F (Submodule.map O E.ker) := by
  let f : E.ker →ₗ[F] Z := O.comp E.ker.subtype
  have hr : f.range = Submodule.map O E.ker := by
    ext z
    constructor
    · rintro ⟨v, rfl⟩
      exact ⟨v, v.property, rfl⟩
    · rintro ⟨v, hv, rfl⟩
      exact ⟨⟨v, hv⟩, rfl⟩
  have hk : Submodule.map E.ker.subtype f.ker = (E.prod O).ker := by
    ext v
    constructor
    · rintro ⟨w, hw, rfl⟩
      exact LinearMap.mem_ker.mpr (Prod.ext (LinearMap.mem_ker.mp w.property)
        (LinearMap.mem_ker.mp hw))
    · intro hv
      have he : E v = 0 := congrArg Prod.fst (LinearMap.mem_ker.mp hv)
      have ho : O v = 0 := congrArg Prod.snd (LinearMap.mem_ker.mp hv)
      exact ⟨⟨v, LinearMap.mem_ker.mpr he⟩, LinearMap.mem_ker.mpr ho, rfl⟩
  have hd : finrank F (E.prod O).ker = finrank F f.ker := by
    rw [← hk, Submodule.finrank_map_subtype_eq]
  have h₁ := E.finrank_range_add_finrank_ker
  have h₂ := (E.prod O).finrank_range_add_finrank_ker
  have h₃ := f.finrank_range_add_finrank_ker
  rw [hr] at h₃
  omega

theorem observed_kernel_eq_candidate_iff_rank [FiniteDimensional F V]
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (Q : T →ₗ[F] V)
    (hQ : Q.range ≤ E.ker) :
    Submodule.map O E.ker = Submodule.map O Q.range ↔
      finrank F (E.prod O).range - finrank F E.range = finrank F (O.comp Q).range := by
  rw [observer_rank_increment, LinearMap.range_comp]
  constructor
  · intro h
    rw [h]
  · intro h
    exact (Submodule.eq_of_le_of_finrank_eq (Submodule.map_mono hQ) h.symm).symm

end ProjectedKernelRigidity
