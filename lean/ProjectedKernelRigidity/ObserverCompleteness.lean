import Mathlib.LinearAlgebra.Basis.VectorSpace
import Mathlib.LinearAlgebra.Isomorphisms

/-!
# Observer-complete kernel certificates

This file proves the exact factor-through criterion used to certify that a
known subspace of a kernel contains every direction visible to an observer.
The full kernel need not be constructed.
-/

namespace ProjectedKernelRigidity

open LinearMap Submodule

variable {F V Y Z T : Type*} [Field F]
variable [AddCommGroup V] [Module F V]
variable [AddCommGroup Y] [Module F Y]
variable [AddCommGroup Z] [Module F Z]
variable [AddCommGroup T] [Module F T]

/-- A checkable certificate that the observer, after quotienting by a declared
visible subspace `R`, factors through the equation map `E`. -/
structure ObserverCompletenessCertificate
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z) where
  factor : Y →ₗ[F] Z ⧸ R
  identity : factor.comp E = R.mkQ.comp O

/-- A coefficient-bearing witness that the declared visible subspace misses
an actually observable kernel direction. -/
structure ObserverEscapeWitness
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z) where
  source : V
  equation : E source = 0
  escapes : O source ∉ R

/-- A factor certificate proves that every kernel vector is observed inside
the declared visible subspace. -/
theorem ObserverCompletenessCertificate.sound
    {E : V →ₗ[F] Y} {O : V →ₗ[F] Z} {R : Submodule F Z}
    (certificate : ObserverCompletenessCertificate E O R) :
    Submodule.map O E.ker ≤ R := by
  rintro z ⟨v, hv, rfl⟩
  have hEv : E v = 0 := LinearMap.mem_ker.mp hv
  have hidentity := LinearMap.congr_fun certificate.identity v
  have hzero : R.mkQ (O v) = 0 := by
    simpa [hEv] using hidentity.symm
  exact (Submodule.Quotient.mk_eq_zero R).mp hzero

/-- If the observer image of the kernel is contained in `R`, a factor
certificate exists. This is the completeness half of the alternative. -/
theorem exists_observerCompletenessCertificate
    {E : V →ₗ[F] Y} {O : V →ₗ[F] Z} {R : Submodule F Z}
    (hvisible : Submodule.map O E.ker ≤ R) :
    Nonempty (ObserverCompletenessCertificate E O R) := by
  let B : V →ₗ[F] Z ⧸ R := R.mkQ.comp O
  have hker : E.ker ≤ B.ker := by
    intro v hv
    rw [LinearMap.mem_ker]
    apply (Submodule.Quotient.mk_eq_zero R).mpr
    apply hvisible
    exact ⟨v, hv, rfl⟩
  let Bq : (V ⧸ E.ker) →ₗ[F] Z ⧸ R := E.ker.liftQ B hker
  let onRange : E.range →ₗ[F] Z ⧸ R :=
    Bq.comp E.quotKerEquivRange.symm.toLinearMap
  obtain ⟨H, hH⟩ := LinearMap.exists_extend onRange
  refine ⟨⟨H, ?_⟩⟩
  ext v
  let ev : E.range := ⟨E v, ⟨v, rfl⟩⟩
  have hHapply := LinearMap.congr_fun hH ev
  have hev : E.quotKerEquivRange.symm ev = E.ker.mkQ v := by
    exact E.quotKerEquivRange_symm_apply_image v ev.property
  change H (E v) = R.mkQ (O v)
  calc
    H (E v) = onRange ev := hHapply
    _ = Bq (E.quotKerEquivRange.symm ev) := rfl
    _ = Bq (E.ker.mkQ v) := by rw [hev]
    _ = B v := Submodule.liftQ_apply _ _ v
    _ = R.mkQ (O v) := rfl

/-- Exact observer-completeness alternative. -/
theorem observerCompletenessCertificate_nonempty_iff
    {E : V →ₗ[F] Y} {O : V →ₗ[F] Z} {R : Submodule F Z} :
    Nonempty (ObserverCompletenessCertificate E O R) ↔
      Submodule.map O E.ker ≤ R := by
  constructor
  · rintro ⟨certificate⟩
    exact certificate.sound
  · exact exists_observerCompletenessCertificate

theorem observerEscapeWitness_nonempty_iff
    {E : V →ₗ[F] Y} {O : V →ₗ[F] Z} {R : Submodule F Z} :
    Nonempty (ObserverEscapeWitness E O R) ↔
      ¬ Submodule.map O E.ker ≤ R := by
  constructor
  · rintro ⟨witness⟩ hle
    apply witness.escapes
    apply hle
    exact ⟨witness.source, LinearMap.mem_ker.mpr witness.equation, rfl⟩
  · intro hnot
    rw [SetLike.not_le_iff_exists] at hnot
    obtain ⟨z, hzmap, hzR⟩ := hnot
    obtain ⟨v, hvker, rfl⟩ := hzmap
    exact ⟨⟨v, LinearMap.mem_ker.mp hvker, hzR⟩⟩

/-- Over a field, observer completeness has an exact proof-producing
alternative: either a factor certificate or an escaping kernel vector. -/
theorem observerCertificate_or_escapeWitness
    (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z) :
    Nonempty (ObserverCompletenessCertificate E O R) ∨
      Nonempty (ObserverEscapeWitness E O R) := by
  by_cases h : Submodule.map O E.ker ≤ R
  · exact Or.inl (exists_observerCompletenessCertificate h)
  · exact Or.inr (observerEscapeWitness_nonempty_iff.mpr h)

/-- If `Q` already consists of exact kernel vectors and its observed image is
observer-complete, then it equals the complete observed kernel image. -/
theorem observed_kernel_eq_candidate
    {E : V →ₗ[F] Y} {O : V →ₗ[F] Z} {Q : T →ₗ[F] V}
    (hQ : LinearMap.range Q ≤ E.ker)
    (hcomplete : Submodule.map O E.ker ≤ Submodule.map O (LinearMap.range Q)) :
    Submodule.map O E.ker = Submodule.map O (LinearMap.range Q) := by
  apply le_antisymm hcomplete
  exact Submodule.map_mono hQ

/-- A certificate with `R` equal to the observed candidate image gives exact
observer completeness without asserting that the candidate is the full
kernel. -/
theorem certificate_proves_observed_kernel_eq_candidate
    {E : V →ₗ[F] Y} {O : V →ₗ[F] Z} {Q : T →ₗ[F] V}
    (hQ : LinearMap.range Q ≤ E.ker)
    (certificate : ObserverCompletenessCertificate E O
      (Submodule.map O (LinearMap.range Q))) :
    Submodule.map O E.ker = Submodule.map O (LinearMap.range Q) := by
  exact observed_kernel_eq_candidate hQ certificate.sound

end ProjectedKernelRigidity
