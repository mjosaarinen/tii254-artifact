import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.LinearAlgebra.Quotient.Basic
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Data.ZMod.Basic

/-!
# Zero-primary reachability and decisive panel certificates

This module isolates the missing implication between a right-start visibility
statement and recovery of vectors in the kernel of an endomorphism.

The main positive theorem is a finite-dimensional Nakayama argument stated
without Jordan form: an invariant reachable space `C` contains `ker A` once
its image surjects onto `V / range A`, equivalently once
`C ⊔ range A = ⊤`.  Thus a Toeplitz visibility bound becomes a complete
zero-primary reachability bound only when it is applied to the *whole*
cokernel of `A`, not merely to an unrelated rank-`g` quotient.

The second group of results formalizes the deterministic alternative.  A
replayed candidate panel whose dimension reaches a certified upper bound on
`dim ker A` is the complete kernel.  Exhaustive absence of an observer profile
inside such a panel is consequently a structural refutation, whereas absence
inside an incomplete panel is only a panel miss.
-/

namespace ProjectedKernelRigidity.ZeroPrimaryReachability

open Submodule

variable {F V W S : Type*} [Field F] [AddCommGroup V] [Module F V]

/-- Surjectivity after a quotient map says exactly that the supplied range,
together with the quotient kernel, spans the ambient space. -/
theorem range_sup_kernel_eq_top_of_surjective_comp
    [AddCommGroup W] [Module F W] [AddCommGroup S] [Module F S]
    (Y : S →ₗ[F] V) (psi : V →ₗ[F] W)
    (hvisibility : Function.Surjective (psi.comp Y)) :
    Y.range ⊔ psi.ker = ⊤ := by
  apply le_antisymm le_top
  intro v _
  obtain ⟨s, hs⟩ := hvisibility (psi v)
  have hs' : psi (Y s) = psi v := by simpa using hs
  have hdiff : v - Y s ∈ psi.ker := by
    rw [LinearMap.mem_ker, map_sub, hs', sub_self]
  have hy : Y s ∈ Y.range := ⟨s, rfl⟩
  have hv : v = Y s + (v - Y s) := by abel
  rw [hv]
  exact (Y.range ⊔ psi.ker).add_mem
    (Submodule.mem_sup_left hy) (Submodule.mem_sup_right hdiff)

/-- The endomorphism induced by `A` on the quotient by an invariant subspace. -/
noncomputable def quotientEndomorphism (A : V →ₗ[F] V) (C : Submodule F V)
    (hC : C ≤ C.comap A) : (V ⧸ C) →ₗ[F] (V ⧸ C) :=
  C.mapQ C A hC

/-- Finite-dimensional Nakayama form of zero-primary reachability.

If `C` is `A`-invariant and is surjective modulo `range A`, then every vector
killed by `A` already lies in `C`. -/
theorem invariant_coker_surjective_contains_kernel [FiniteDimensional F V]
    (A : V →ₗ[F] V) (C : Submodule F V) (hC : C ≤ C.comap A)
    (hspan : C ⊔ A.range = ⊤) : A.ker ≤ C := by
  let Aq : (V ⧸ C) →ₗ[F] (V ⧸ C) := quotientEndomorphism A C hC
  have hsurj : Function.Surjective Aq := by
    intro z
    obtain ⟨v, rfl⟩ := C.mkQ_surjective z
    have hv : v ∈ C ⊔ A.range := by rw [hspan]; exact Submodule.mem_top
    obtain ⟨c, hc, r, hr, hcr⟩ := Submodule.mem_sup.mp hv
    obtain ⟨y, rfl⟩ := hr
    refine ⟨C.mkQ y, ?_⟩
    change (C.mapQ C A hC) (C.mkQ y) = C.mkQ v
    rw [Submodule.mkQ_apply, Submodule.mapQ_apply]
    apply (Submodule.Quotient.eq C).2
    have heq : A y - v = -c := by rw [← hcr]; abel
    rw [heq]
    exact C.neg_mem hc
  have hinj : Function.Injective Aq :=
    LinearMap.injective_iff_surjective.mpr hsurj
  intro x hx
  have hzero : Aq (C.mkQ x) = 0 := by
    change (C.mapQ C A hC) (C.mkQ x) = 0
    rw [Submodule.mkQ_apply, Submodule.mapQ_apply]
    rw [LinearMap.mem_ker.mp hx]
    exact map_zero C.mkQ
  have hquot : C.mkQ x = 0 := by
    apply hinj
    simpa only [map_zero] using hzero
  exact (Submodule.Quotient.mk_eq_zero C).mp hquot

/-- Right-start form of the reachability theorem.  Any invariant space that
contains the right-start image contains `ker A` when the right start and
`range A` span the ambient space. -/
theorem right_start_coker_visibility_contains_kernel [FiniteDimensional F V]
    (A : V →ₗ[F] V) {YSpace C : Submodule F V}
    (hY : YSpace ≤ C) (hC : C ≤ C.comap A)
    (hvisibility : YSpace ⊔ A.range = ⊤) : A.ker ≤ C := by
  apply invariant_coker_surjective_contains_kernel A C hC
  apply le_antisymm le_top
  calc
    ⊤ = YSpace ⊔ A.range := hvisibility.symm
    _ ≤ C ⊔ A.range := sup_le_sup hY le_rfl

/-- Matrix-free bridge from a full cokernel visibility test to zero-primary
reachability.  The load-bearing hypothesis is `ker psi = range A`: replacing
`psi` by an arbitrary lower-rank quotient does not justify this conclusion. -/
theorem full_coker_visibility_contains_kernel [FiniteDimensional F V]
    [AddCommGroup W] [Module F W] [AddCommGroup S] [Module F S]
    (A : V →ₗ[F] V) (Y : S →ₗ[F] V) (psi : V →ₗ[F] W)
    (C : Submodule F V) (hY : Y.range ≤ C) (hC : C ≤ C.comap A)
    (hpsi : psi.ker = A.range)
    (hvisibility : Function.Surjective (psi.comp Y)) : A.ker ≤ C := by
  apply right_start_coker_visibility_contains_kernel A hY hC
  rw [← hpsi]
  exact range_sup_kernel_eq_top_of_surjective_comp Y psi hvisibility

/-- A shifted block-Hankel matrix factors through one application of `A`, so
its rank is a lower bound on `rank A`.  This statement is false for an
unshifted factor `O*C` when `A = 0`; the explicit `A` factor is load-bearing. -/
theorem shifted_hankel_rank_le_operator {L N R : Type*}
    [Fintype L] [Fintype N] [Fintype R]
    (O : Matrix L N F) (A : Matrix N N F) (C : Matrix N R F) :
    (O * A * C).rank ≤ A.rank := by
  exact (Matrix.rank_mul_le_left (O * A) C).trans
    (Matrix.rank_mul_le_right O A)

/-- A replayed panel is the complete kernel when its certified dimension
reaches any valid upper bound on the kernel dimension. -/
theorem replayed_panel_eq_kernel_of_upper_bound [FiniteDimensional F V]
    (A : V →ₗ[F] V) (Q : Submodule F V) (q upper : ℕ)
    (hreplay : Q ≤ A.ker) (hQ : Module.finrank F Q = q)
    (hupper : Module.finrank F A.ker ≤ upper) (hmatch : q = upper) :
    Q = A.ker := by
  apply Submodule.eq_of_le_of_finrank_le hreplay
  simpa [hQ, hmatch] using hupper

/-- The observer profile used by an anchored panel: `G` has the requested
dimension, is killed by the anchor observer, and meets every other observer
kernel trivially.  The last condition is exactly injectivity of each observer
on `G`. -/
def ObserverAnchorProfile [AddCommGroup W] [Module F W] {I : Type*}
    (anchor : V →ₗ[F] W) (observers : I → V →ₗ[F] W) (g : ℕ)
    (G : Submodule F V) : Prop :=
  Module.finrank F G = g ∧ G ≤ anchor.ker ∧
    ∀ i, Disjoint G (observers i).ker

/-- Once a panel is certified equal to `ker A`, searching that panel for an
observer profile is logically equivalent to searching the whole kernel. -/
theorem complete_panel_profile_iff [FiniteDimensional F V]
    [AddCommGroup W] [Module F W] {I : Type*}
    (A : V →ₗ[F] V) (Q : Submodule F V) (hcomplete : Q = A.ker)
    (anchor : V →ₗ[F] W) (observers : I → V →ₗ[F] W) (g : ℕ) :
    (∃ G : Submodule F V, G ≤ Q ∧ ObserverAnchorProfile anchor observers g G) ↔
      ∃ G : Submodule F V, G ≤ A.ker ∧ ObserverAnchorProfile anchor observers g G := by
  rw [hcomplete]

/-- Negative form of the decisive-panel theorem. -/
theorem no_profile_in_complete_panel_refutes_kernel_profile [FiniteDimensional F V]
    [AddCommGroup W] [Module F W] {I : Type*}
    (A : V →ₗ[F] V) (Q : Submodule F V) (hcomplete : Q = A.ker)
    (anchor : V →ₗ[F] W) (observers : I → V →ₗ[F] W) (g : ℕ)
    (hnone : ¬ ∃ G : Submodule F V,
      G ≤ Q ∧ ObserverAnchorProfile anchor observers g G) :
    ¬ ∃ G : Submodule F V,
      G ≤ A.ker ∧ ObserverAnchorProfile anchor observers g G := by
  simpa only [complete_panel_profile_iff A Q hcomplete anchor observers g] using hnone

/-! ## A small counterexample to the weaker quotient claim -/

/-- A fixed quotient can see a right start at full rank even though the
right-start space does not contain the selected kernel line.  Here `A = 0`,
`Y = (1,1)^T`, `psi = (1,0)`, and the desired kernel vector is `(1,0)^T`.
This is why a rank-`g` quotient must be tied to the full cokernel or supplied
with an additional lifting/splitting theorem. -/
example :
    let A : Matrix (Fin 2) (Fin 2) (ZMod 2) := !![0, 0; 0, 0]
    let Y : Matrix (Fin 2) (Fin 1) (ZMod 2) := !![1; 1]
    let psi : Matrix (Fin 1) (Fin 2) (ZMod 2) := !![1, 0]
    let desired : Fin 2 → ZMod 2 := ![1, 0]
    psi * A = 0 ∧ psi * Y = 1 ∧ A.mulVec desired = 0 ∧
      ¬ ∃ c : Fin 1 → ZMod 2, Y.mulVec c = desired := by
  decide

end ProjectedKernelRigidity.ZeroPrimaryReachability
