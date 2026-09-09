import Mathlib.LinearAlgebra.Dual.Lemmas
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.LinearAlgebra.Quotient.Basic
import Mathlib.LinearAlgebra.Prod
import Mathlib.LinearAlgebra.Span.Basic
import Mathlib.Order.ModularLattice
import Mathlib.Order.SupIndep
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Finset.Card
import Mathlib.LinearAlgebra.Vandermonde
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.LinearAlgebra.Dimension.RankNullity
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.LinearPMap
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.FinCases

/-!
# Two-anchor pair-core recovery: the reusable linear algebra

This module formalizes the finite-dimensional core of the solution report
`two_anchor_pair_core_solution.md`.  Everything is stated over an arbitrary field
`K` (the application has `K = GF(2)`) and, wherever no dimension count is needed,
over an arbitrary ring.

Notation.  `Kk : ι → Submodule K V` is the family of complete observer kernels
inside the ambient space `V` (called `S` in the problem statement),
`rowSpace Kk j = (Kk j).dualAnnihilator` is the observer row space `R_j ≤ V*`,
`pairCore Kk = ⨅_{i ≠ j} (K_i ⊔ K_j)` is the pair-core extractor `P` of (6), and
`sharedNuisance Kk = ⨆_{i ≠ j} (R_i ⊓ R_j)`.

Results.

* **Theorem target B** — `dualAnnihilator_pairCore` is the exact dual identity (7)
  `P^⊥ = Σ_{i≠j} (R_i ∩ R_j)`; `pairCore_eq_iff` is the certificate (8)
  `P = C₀ ↔ Σ (R_i ∩ R_j) = C₀^⊥` (no hypothesis is needed for the equivalence);
  `le_pairCore_iff_dual`/`pairCore_le_iff_dual` split it into the two inclusions;
  `pairCore_eq_of_witness` is the economical certificate: finitely many covectors,
  each lying in some `R_i ∩ R_j`, whose span contains `C₀^⊥`.
* **Shear invariance** — `inf_eq_inf_of_disjoint_map` and
  `sharedNuisance_eq_of_disjoint_restrict`: if the restrictions of `R_i` and `R_j`
  to `C₀` are disjoint (the canonical `m/m/2m` geometry), then
  `R_i ∩ R_j = M_i ∩ M_j` with `M_j = R_j ∩ C₀^⊥` the nuisance actually seen by
  observer `j`; so `P = C₀ ↔ Σ (M_i ∩ M_j) = C₀^⊥` (`pairCore_eq_iff_nuisance`).
  No direct-sum decomposition of the ambient dual is used.
* **Theorem target C** — `sup_inf_sup_of_iSupIndep` (independent families in a
  modular lattice: `(⨆_A N) ⊓ (⨆_B N) = ⨆_{A∩B} N`), `nuisancePart_inf` (identity
  (11)) and `pairCore_eq_iff_support` (criterion (12)): in the support-hypergraph
  model, `P = C₀` iff every nonzero nuisance block is seen by at least two observers.
-/

namespace TwoAnchorPairCore

open Module Submodule

/-! ### The pair core and its exact dual certificate -/

section DualCertificate

variable {K V : Type*} [Field K] [AddCommGroup V] [Module K V] {ι : Type*}

/-- Ordered pairs of distinct observers (each unordered pair appears twice; harmless). -/
abbrev Pairs (ι : Type*) := {p : ι × ι // p.1 ≠ p.2}

/-- The pair-core extractor `P = ⨅_{i ≠ j} (K_i ⊔ K_j)` of (6). -/
def pairCore (Kk : ι → Submodule K V) : Submodule K V :=
  ⨅ p : Pairs ι, Kk p.1.1 ⊔ Kk p.1.2

/-- Observer row space `R_j = K_j^⊥ ≤ V*`. -/
def rowSpace (Kk : ι → Submodule K V) (j : ι) : Submodule K (Dual K V) :=
  (Kk j).dualAnnihilator

/-- `Σ_{i ≠ j} (R_i ⊓ R_j)`. -/
def sharedNuisance (Kk : ι → Submodule K V) : Submodule K (Dual K V) :=
  ⨆ p : Pairs ι, rowSpace Kk p.1.1 ⊓ rowSpace Kk p.1.2

theorem mem_pairCore {Kk : ι → Submodule K V} {x : V} :
    x ∈ pairCore Kk ↔ ∀ i j, i ≠ j → x ∈ Kk i ⊔ Kk j := by
  simp [pairCore, Submodule.mem_iInf, Subtype.forall, Prod.forall]

theorem le_pairCore_iff {Kk : ι → Submodule K V} {C0 : Submodule K V} :
    C0 ≤ pairCore Kk ↔ ∀ i j, i ≠ j → C0 ≤ Kk i ⊔ Kk j := by
  simp [pairCore, le_iInf_iff, Subtype.forall, Prod.forall]

/-- **Theorem target B, identity (7)**: `P^⊥ = Σ_{i≠j} (R_i ⊓ R_j)`. -/
theorem dualAnnihilator_pairCore [FiniteDimensional K V] [Finite ι]
    (Kk : ι → Submodule K V) :
    (pairCore Kk).dualAnnihilator = sharedNuisance Kk := by
  unfold pairCore sharedNuisance rowSpace
  rw [Subspace.dualAnnihilator_iInf_eq]
  simp only [Submodule.dualAnnihilator_sup_eq]

/-- **Theorem target B, certificate (8)**: `P = C₀ ↔ Σ (R_i ⊓ R_j) = C₀^⊥`.
No inclusion hypothesis is needed for the equivalence itself. -/
theorem pairCore_eq_iff [FiniteDimensional K V] [Finite ι]
    (Kk : ι → Submodule K V) (C0 : Submodule K V) :
    pairCore Kk = C0 ↔ sharedNuisance Kk = C0.dualAnnihilator := by
  rw [← Subspace.dualAnnihilator_inj, dualAnnihilator_pairCore]

/-- The inclusion `C₀ ≤ P` (equivalently `C₀ ≤ K_i ⊔ K_j` for all pairs) in dual form. -/
theorem le_pairCore_iff_dual [FiniteDimensional K V] [Finite ι]
    {Kk : ι → Submodule K V} {C0 : Submodule K V} :
    C0 ≤ pairCore Kk ↔ sharedNuisance Kk ≤ C0.dualAnnihilator := by
  rw [← Subspace.dualAnnihilator_le_dualAnnihilator_iff, dualAnnihilator_pairCore]

/-- The inclusion `P ≤ C₀` in dual form. -/
theorem pairCore_le_iff_dual [FiniteDimensional K V] [Finite ι]
    {Kk : ι → Submodule K V} {C0 : Submodule K V} :
    pairCore Kk ≤ C0 ↔ C0.dualAnnihilator ≤ sharedNuisance Kk := by
  rw [← Subspace.dualAnnihilator_le_dualAnnihilator_iff, dualAnnihilator_pairCore]

/-- Any covector lying in some `R_i ⊓ R_j` lies in `Σ (R_i ⊓ R_j)`. -/
theorem mem_sharedNuisance_of_mem {Kk : ι → Submodule K V} {x : Dual K V} {i j : ι}
    (hij : i ≠ j) (hx : x ∈ rowSpace Kk i ⊓ rowSpace Kk j) : x ∈ sharedNuisance Kk :=
  (le_iSup (fun p : Pairs ι => rowSpace Kk p.1.1 ⊓ rowSpace Kk p.1.2) ⟨(i, j), hij⟩) hx

/-- **The economical deterministic certificate.**  If `C₀ ≤ K_i ⊔ K_j` for all
pairs (the structural half, e.g. from the canonical geometry) and a finite set `s`
of covectors, each lying in some `R_i ⊓ R_j`, spans a space containing `C₀^⊥`,
then `P = C₀`.  The verifier checks membership of each witness in two row spaces
and one rank. -/
theorem pairCore_eq_of_witness [FiniteDimensional K V] [Finite ι]
    {Kk : ι → Submodule K V} {C0 : Submodule K V}
    (hC : ∀ i j, i ≠ j → C0 ≤ Kk i ⊔ Kk j) (s : Set (Dual K V))
    (hs : ∀ x ∈ s, ∃ i j, i ≠ j ∧ x ∈ rowSpace Kk i ⊓ rowSpace Kk j)
    (hspan : C0.dualAnnihilator ≤ span K s) : pairCore Kk = C0 := by
  refine le_antisymm ?_ (le_pairCore_iff.mpr hC)
  rw [pairCore_le_iff_dual]
  refine hspan.trans (span_le.mpr ?_)
  intro x hx
  obtain ⟨i, j, hij, hx⟩ := hs x hx
  exact mem_sharedNuisance_of_mem hij hx

end DualCertificate

/-! ### Shear invariance: the intersection only sees the nuisance parts -/

section Shear

variable {R M N : Type*} [Ring R] [AddCommGroup M] [Module R M] [AddCommGroup N] [Module R N]

/-- If the images of `A` and `B` under `f` are disjoint then `A ⊓ B ≤ ker f`. -/
theorem inf_le_ker_of_disjoint_map (f : M →ₗ[R] N) {A B : Submodule R M}
    (h : Disjoint (A.map f) (B.map f)) : A ⊓ B ≤ LinearMap.ker f := by
  intro x hx
  have hx' : f x ∈ A.map f ⊓ B.map f := ⟨⟨x, hx.1, rfl⟩, ⟨x, hx.2, rfl⟩⟩
  rw [disjoint_iff.mp h] at hx'
  simpa [LinearMap.mem_ker] using hx'

/-- **Shear invariance.**  If the images of `A` and `B` under `f` are disjoint then
`A ⊓ B = (A ⊓ ker f) ⊓ (B ⊓ ker f)`. -/
theorem inf_eq_inf_of_disjoint_map (f : M →ₗ[R] N) {A B : Submodule R M}
    (h : Disjoint (A.map f) (B.map f)) :
    A ⊓ B = (A ⊓ LinearMap.ker f) ⊓ (B ⊓ LinearMap.ker f) := by
  refine le_antisymm ?_ (inf_le_inf inf_le_left inf_le_left)
  intro x hx
  exact ⟨⟨hx.1, inf_le_ker_of_disjoint_map f h hx⟩, ⟨hx.2, inf_le_ker_of_disjoint_map f h hx⟩⟩

/-- Disjoint subspaces whose sum meets `ker f` trivially have disjoint images. -/
theorem disjoint_map_of_disjoint (f : M →ₗ[R] N) {A B : Submodule R M}
    (hAB : Disjoint A B) (hker : Disjoint (A ⊔ B) (LinearMap.ker f)) :
    Disjoint (A.map f) (B.map f) := by
  rw [Submodule.disjoint_def]
  rintro _ ⟨a, ha, rfl⟩ ⟨b, hb, hab⟩
  have h1 : a - b ∈ A ⊔ B := Submodule.sub_mem _ (mem_sup_left ha) (mem_sup_right hb)
  have h2 : a - b ∈ LinearMap.ker f := by
    rw [LinearMap.mem_ker, map_sub, hab, sub_self]
  have h3 : a - b = 0 := (Submodule.disjoint_def.mp hker) _ h1 h2
  have h4 : a = b := sub_eq_zero.mp h3
  subst h4
  have : a = 0 := (Submodule.disjoint_def.mp hAB) a ha hb
  simp [this]

/-- If `A ≤ W` then `(D ⊔ A) ⊓ W = A ⊔ (D ⊓ W)`; in particular `= A` when `D ⊓ W = ⊥`. -/
theorem sup_inf_of_le {α : Type*} [Lattice α] [IsModularLattice α] {A D W : α} (hA : A ≤ W) :
    (D ⊔ A) ⊓ W = A ⊔ (D ⊓ W) := by
  rw [sup_comm]
  exact sup_inf_assoc_of_le D hA

end Shear

section ShearApplied

variable {K V : Type*} [Field K] [AddCommGroup V] [Module K V] {ι : Type*}

/-- The canonical part of observer `j`: the restriction of its row space to `C₀`
(a subspace of `C₀*`, the projected-conic subspace `D_j` in the application). -/
def canonicalPart (Kk : ι → Submodule K V) (C0 : Submodule K V) (j : ι) :
    Submodule K (Dual K C0) := (rowSpace Kk j).map C0.dualRestrict

/-- The nuisance part of observer `j`: `M_j = R_j ⊓ C₀^⊥`. -/
def nuisanceSeen (Kk : ι → Submodule K V) (C0 : Submodule K V) (j : ι) :
    Submodule K (Dual K V) := rowSpace Kk j ⊓ C0.dualAnnihilator

/-- **Shear-invariant pair intersection.**  If the canonical parts of two observers
are disjoint, `R_i ⊓ R_j = M_i ⊓ M_j`. -/
theorem rowSpace_inf_eq_nuisanceSeen_inf {Kk : ι → Submodule K V} {C0 : Submodule K V}
    {i j : ι} (h : Disjoint (canonicalPart Kk C0 i) (canonicalPart Kk C0 j)) :
    rowSpace Kk i ⊓ rowSpace Kk j = nuisanceSeen Kk C0 i ⊓ nuisanceSeen Kk C0 j := by
  unfold nuisanceSeen
  rw [← C0.dualRestrict_ker_eq_dualAnnihilator]
  exact inf_eq_inf_of_disjoint_map _ h

theorem sharedNuisance_eq_of_disjoint_restrict {Kk : ι → Submodule K V} {C0 : Submodule K V}
    (h : ∀ i j, i ≠ j → Disjoint (canonicalPart Kk C0 i) (canonicalPart Kk C0 j)) :
    sharedNuisance Kk = ⨆ p : Pairs ι, nuisanceSeen Kk C0 p.1.1 ⊓ nuisanceSeen Kk C0 p.1.2 := by
  unfold sharedNuisance
  exact iSup_congr fun p => rowSpace_inf_eq_nuisanceSeen_inf (h _ _ p.2)

/-- **The general exact criterion.**  Under pairwise-disjoint canonical parts,
`P = C₀ ↔ Σ_{i≠j} (M_i ⊓ M_j) = C₀^⊥`. -/
theorem pairCore_eq_iff_nuisance [FiniteDimensional K V] [Finite ι]
    {Kk : ι → Submodule K V} {C0 : Submodule K V}
    (h : ∀ i j, i ≠ j → Disjoint (canonicalPart Kk C0 i) (canonicalPart Kk C0 j)) :
    pairCore Kk = C0 ↔
      (⨆ p : Pairs ι, nuisanceSeen Kk C0 p.1.1 ⊓ nuisanceSeen Kk C0 p.1.2) = C0.dualAnnihilator := by
  rw [pairCore_eq_iff, sharedNuisance_eq_of_disjoint_restrict h]

/-- Under pairwise-disjoint canonical parts the structural inclusion `C₀ ≤ P` is automatic. -/
theorem le_pairCore_of_disjoint_restrict [FiniteDimensional K V] [Finite ι]
    {Kk : ι → Submodule K V} {C0 : Submodule K V}
    (h : ∀ i j, i ≠ j → Disjoint (canonicalPart Kk C0 i) (canonicalPart Kk C0 j)) :
    C0 ≤ pairCore Kk := by
  rw [le_pairCore_iff_dual, sharedNuisance_eq_of_disjoint_restrict h]
  exact iSup_le fun p => inf_le_left.trans inf_le_right

end ShearApplied

/-! ### Independent families in a modular lattice -/

section Independent

variable {α : Type*} [CompleteLattice α] [IsModularLattice α] {τ : Type*} [DecidableEq τ]
  {t : τ → α}

/-- Disjoint index sets of an independent family have disjoint partial sums. -/
theorem disjoint_sup_sup_of_iSupIndep (ht : iSupIndep t) (A B : Finset τ)
    (hAB : Disjoint A B) : Disjoint (A.sup t) (B.sup t) := by
  induction A using Finset.induction_on with
  | empty => simp
  | insert a A ha ih =>
    rw [Finset.sup_insert]
    have hB : Disjoint A B := hAB.mono_left (Finset.subset_insert a A)
    have haB : a ∉ B := Finset.disjoint_left.mp hAB (Finset.mem_insert_self a A)
    have h1 : Disjoint (t a) (A.sup t ⊔ B.sup t) := by
      rw [← Finset.sup_union]
      have hnot : a ∉ ((A ∪ B : Finset τ) : Set τ) := by simpa using not_or.mpr ⟨ha, haB⟩
      refine (ht.disjoint_biSup hnot).mono_right ?_
      exact Finset.sup_le fun i hi =>
        le_iSup₂ (f := fun i (_ : i ∈ ((A ∪ B : Finset τ) : Set τ)) => t i) i (by simpa using hi)
    exact (ih hB).disjoint_sup_left_of_disjoint_sup_right h1

/-- **Partial sums of an independent family intersect along the common index set.** -/
theorem sup_inf_sup_of_iSupIndep (ht : iSupIndep t) (A B : Finset τ) :
    A.sup t ⊓ B.sup t = (A ∩ B).sup t := by
  have hdisj : Disjoint ((A \ B).sup t) (B.sup t) :=
    disjoint_sup_sup_of_iSupIndep ht _ _ Finset.sdiff_disjoint
  calc A.sup t ⊓ B.sup t
      = ((A ∩ B).sup t ⊔ (A \ B).sup t) ⊓ B.sup t := by
        rw [← Finset.sup_union, Finset.union_comm, Finset.sdiff_union_inter]
    _ = (A ∩ B).sup t ⊔ ((A \ B).sup t ⊓ B.sup t) :=
        sup_inf_assoc_of_le _ (Finset.sup_mono Finset.inter_subset_right)
    _ = (A ∩ B).sup t := by rw [disjoint_iff.mp hdisj, sup_bot_eq]

end Independent

/-! ### The support-hypergraph model (Theorem target C) -/

section Hypergraph

variable {K V : Type*} [Field K] [AddCommGroup V] [Module K V]
  {ι τ : Type*} [Fintype τ] [DecidableEq τ] [DecidableEq ι]

/-- The blocks seen by observer `j`. -/
def blocksOf (supp : τ → Finset ι) (j : ι) : Finset τ :=
  Finset.univ.filter fun b => j ∈ supp b

/-- The nuisance part `M_j = ⨆_{T ∋ j} N_T` of observer `j` in the support model (10). -/
def nuisancePart (N : τ → Submodule K (Dual K V)) (supp : τ → Finset ι) (j : ι) :
    Submodule K (Dual K V) := (blocksOf supp j).sup N

/-- **Identity (11)**: `M_i ⊓ M_j = ⨆_{T ⊇ {i,j}} N_T` for independent blocks. -/
theorem nuisancePart_inf {N : τ → Submodule K (Dual K V)} (hN : iSupIndep N)
    (supp : τ → Finset ι) (i j : ι) :
    nuisancePart N supp i ⊓ nuisancePart N supp j =
      (Finset.univ.filter fun b => i ∈ supp b ∧ j ∈ supp b).sup N := by
  unfold nuisancePart
  rw [sup_inf_sup_of_iSupIndep hN]
  congr 1
  ext b
  simp [blocksOf]

/-- The total nuisance `N* = ⨆_T N_T`. -/
def totalNuisance (N : τ → Submodule K (Dual K V)) : Submodule K (Dual K V) :=
  Finset.univ.sup N

omit [DecidableEq τ] in
/-- In the support model the observer row space is `R_j = D_j ⊔ M_j` (10); if the
`D_j` are pairwise disjoint and `(D_i ⊔ D_j) ⊓ N* = ⊥` (directness of (9)), then
`R_i ⊓ R_j = M_i ⊓ M_j`. -/
theorem rowSpace_inf_of_support {Kk : ι → Submodule K V} {D : ι → Submodule K (Dual K V)}
    {N : τ → Submodule K (Dual K V)} {supp : τ → Finset ι}
    (hR : ∀ j, rowSpace Kk j = D j ⊔ nuisancePart N supp j)
    (hD : ∀ i j, i ≠ j → Disjoint (D i) (D j))
    (hDN : ∀ i j, i ≠ j → Disjoint (D i ⊔ D j) (totalNuisance N))
    {i j : ι} (hij : i ≠ j) :
    rowSpace Kk i ⊓ rowSpace Kk j = nuisancePart N supp i ⊓ nuisancePart N supp j := by
  set W := totalNuisance N with hW
  have hM : ∀ l, nuisancePart N supp l ≤ W := fun l =>
    Finset.sup_mono (Finset.subset_univ _)
  have hmap : ∀ l, (rowSpace Kk l).map W.mkQ = (D l).map W.mkQ := by
    intro l
    rw [hR l, Submodule.map_sup]
    have : (nuisancePart N supp l).map W.mkQ = ⊥ := by
      rw [← le_bot_iff, ← W.mkQ_map_self]
      exact Submodule.map_mono (hM l)
    rw [this, sup_bot_eq]
  have hdisj : Disjoint ((rowSpace Kk i).map W.mkQ) ((rowSpace Kk j).map W.mkQ) := by
    rw [hmap i, hmap j]
    refine disjoint_map_of_disjoint _ (hD i j hij) ?_
    rw [Submodule.ker_mkQ]
    exact hDN i j hij
  rw [inf_eq_inf_of_disjoint_map _ hdisj, Submodule.ker_mkQ]
  have hDi : Disjoint (D i) W := (hDN i j hij).mono_left le_sup_left
  have hDj : Disjoint (D j) W := (hDN i j hij).mono_left le_sup_right
  rw [hR i, hR j, sup_inf_of_le (hM i), sup_inf_of_le (hM j),
    disjoint_iff.mp hDi, disjoint_iff.mp hDj, sup_bot_eq, sup_bot_eq]

/-- **Criterion (12).**  In the support model with independent blocks, the pair core
equals `C₀` iff every nonzero block is seen by at least two observers. -/
theorem pairCore_eq_iff_support [FiniteDimensional K V] [Finite ι]
    {Kk : ι → Submodule K V} {C0 : Submodule K V} {D : ι → Submodule K (Dual K V)}
    {N : τ → Submodule K (Dual K V)} {supp : τ → Finset ι}
    (hN : iSupIndep N)
    (hR : ∀ j, rowSpace Kk j = D j ⊔ nuisancePart N supp j)
    (hD : ∀ i j, i ≠ j → Disjoint (D i) (D j))
    (hDN : ∀ i j, i ≠ j → Disjoint (D i ⊔ D j) (totalNuisance N))
    (hC0 : C0.dualAnnihilator = totalNuisance N) :
    pairCore Kk = C0 ↔ ∀ b, N b ≠ ⊥ → 2 ≤ (supp b).card := by
  rw [pairCore_eq_iff, hC0]
  have hshared : sharedNuisance Kk =
      ⨆ p : Pairs ι, (Finset.univ.filter fun b => p.1.1 ∈ supp b ∧ p.1.2 ∈ supp b).sup N := by
    unfold sharedNuisance
    exact iSup_congr fun p => by
      rw [rowSpace_inf_of_support hR hD hDN p.2, nuisancePart_inf hN]
  rw [hshared]
  constructor
  · intro heq b hb
    by_contra hcard
    have hcard' : (supp b).card < 2 := not_le.mp hcard
    -- every pair summand avoids block `b`
    have hle : (⨆ p : Pairs ι,
        (Finset.univ.filter fun c => p.1.1 ∈ supp c ∧ p.1.2 ∈ supp c).sup N) ≤
        (Finset.univ.erase b).sup N := by
      refine iSup_le fun p => Finset.sup_mono ?_
      intro c hc
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hc
      rw [Finset.mem_erase]
      refine ⟨?_, Finset.mem_univ _⟩
      rintro rfl
      have : 2 ≤ (supp c).card := by
        rw [Nat.succ_le_iff, Finset.one_lt_card]
        exact ⟨_, hc.1, _, hc.2, p.2⟩
      omega
    have hb' : N b ≤ (Finset.univ.erase b).sup N := by
      calc N b ≤ totalNuisance N := Finset.le_sup (Finset.mem_univ b)
        _ = _ := heq.symm
        _ ≤ _ := hle
    have hdisj : Disjoint (N b) ((Finset.univ.erase b).sup N) := by
      refine (hN.disjoint_biSup (y := ((Finset.univ.erase b : Finset τ) : Set τ))
        (by simp)).mono_right ?_
      exact Finset.sup_le fun c hc =>
        le_iSup₂ (f := fun c (_ : c ∈ ((Finset.univ.erase b : Finset τ) : Set τ)) => N c) c
          (by simpa using hc)
    exact hb (disjoint_self.mp (hdisj.mono_right hb'))
  · intro hcard
    refine le_antisymm (iSup_le fun p => Finset.sup_mono (Finset.subset_univ _)) ?_
    refine Finset.sup_le fun b _ => ?_
    by_cases hb : N b = ⊥
    · rw [hb]; exact bot_le
    · obtain ⟨i, hi, j, hj, hij⟩ := Finset.one_lt_card.mp (Nat.succ_le_iff.mp (hcard b hb))
      refine le_trans ?_ (le_iSup (fun p : Pairs ι =>
        (Finset.univ.filter fun c => p.1.1 ∈ supp c ∧ p.1.2 ∈ supp c).sup N) ⟨(i, j), hij⟩)
      exact Finset.le_sup (by simp [hi, hj])

end Hypergraph

/-! ### The projected-conic geometry (Theorem target A) -/

section Conic

variable (K : Type*) {E : Type*} [Field K] [Field E] [Algebra K E]

/-- The `K`-linear map `s ↦ (s, s u, s u²)` into `U = E³` (written `Fin 3 → E`). -/
def conicMap (u : E) : E →ₗ[K] (Fin 3 → E) where
  toFun s := fun i => s * u ^ (i : ℕ)
  map_add' s t := by ext i; simp [add_mul]
  map_smul' c s := by ext i; simp [Algebra.smul_def, mul_assoc]

theorem conicMap_apply (u s : E) (i : Fin 3) : conicMap K u s i = s * u ^ (i : ℕ) := rfl

/-- The `E`-line `L_u = E·(1,u,u²)` of (1), viewed as a `K`-subspace of `U`. -/
def conicLine (u : E) : Submodule K (Fin 3 → E) := LinearMap.range (conicMap K u)

theorem conicMap_injective (u : E) : Function.Injective (conicMap K u) := by
  intro s t h
  have := congrFun h 0
  simpa [conicMap_apply] using this

theorem mem_conicLine {u : E} {x : Fin 3 → E} :
    x ∈ conicLine K u ↔ ∃ s : E, ∀ i, x i = s * u ^ (i : ℕ) := by
  simp only [conicLine, LinearMap.mem_range]
  constructor
  · rintro ⟨s, rfl⟩; exact ⟨s, fun i => rfl⟩
  · rintro ⟨s, hs⟩; exact ⟨s, funext fun i => (hs i).symm⟩

theorem finrank_conicLine [FiniteDimensional K E] (u : E) :
    Module.finrank K (conicLine K u) = Module.finrank K E :=
  LinearMap.finrank_range_of_inj (conicMap_injective K u)

/-- `L_u ∩ L_v = 0` for `u ≠ v`. -/
theorem disjoint_conicLine {u v : E} (huv : u ≠ v) :
    Disjoint (conicLine K u) (conicLine K v) := by
  rw [Submodule.disjoint_def]
  intro x hx hx'
  obtain ⟨s, hs⟩ := (mem_conicLine K).mp hx
  obtain ⟨t, ht⟩ := (mem_conicLine K).mp hx'
  have h0 : s = t := by
    have h1 := hs 0; have h2 := ht 0
    simp only [Fin.val_zero, pow_zero, mul_one] at h1 h2
    rw [← h1, ← h2]
  have h1 : s * u = s * v := by
    have h1 := hs 1; have h2 := ht 1
    simp only [Fin.val_one, pow_one] at h1 h2
    rw [← h1, h2, h0]
  have hs0 : s = 0 := by
    have : s * (u - v) = 0 := by rw [mul_sub, h1, sub_self]
    rcases mul_eq_zero.mp this with h | h
    · exact h
    · exact absurd (sub_eq_zero.mp h) huv
  funext i
  rw [hs i, hs0, zero_mul]
  rfl

theorem mem_conicLine_sup {u v : E} {x : Fin 3 → E} :
    x ∈ conicLine K u ⊔ conicLine K v ↔
      ∃ s t : E, ∀ i, x i = s * u ^ (i : ℕ) + t * v ^ (i : ℕ) := by
  rw [Submodule.mem_sup]
  constructor
  · rintro ⟨y, hy, z, hz, rfl⟩
    obtain ⟨s, hs⟩ := (mem_conicLine K).mp hy
    obtain ⟨t, ht⟩ := (mem_conicLine K).mp hz
    exact ⟨s, t, fun i => by simp [hs i, ht i]⟩
  · rintro ⟨s, t, h⟩
    refine ⟨conicMap K u s, LinearMap.mem_range.mpr ⟨s, rfl⟩,
      conicMap K v t, LinearMap.mem_range.mpr ⟨t, rfl⟩, ?_⟩
    funext i
    simp [conicMap_apply, h i]

/-- The vector `h = (0, 0, η)`. -/
def hvec (η : E) : Fin 3 → E := ![0, 0, η]

theorem hvec_ne_zero {η : E} (hη : η ≠ 0) : hvec η ≠ 0 := by
  intro h
  have := congrFun h 2
  simp [hvec] at this
  exact hη this

/-- `h ∉ L_u`. -/
theorem hvec_not_mem_conicLine {u η : E} (hη : η ≠ 0) : hvec η ∉ conicLine K u := by
  intro hmem
  obtain ⟨s, hs⟩ := (mem_conicLine K).mp hmem
  have h0 := hs 0
  have h2 := hs 2
  simp [hvec] at h0 h2
  rw [← h0] at h2
  simp at h2
  exact hη h2

/-- `h ∉ L_u + L_v` for `u ≠ v`: the key fact behind the projected pair geometry. -/
theorem hvec_not_mem_sup {u v η : E} (huv : u ≠ v) (hη : η ≠ 0) :
    hvec η ∉ conicLine K u ⊔ conicLine K v := by
  intro hmem
  obtain ⟨s, t, h⟩ := (mem_conicLine_sup K).mp hmem
  have h0 := h 0
  have h1 := h 1
  have h2 := h 2
  simp [hvec] at h0 h1 h2
  have ht : t = -s := by linear_combination -h0
  have hsu : s * (u - v) = 0 := by
    rw [ht] at h1
    linear_combination -h1
  have hs0 : s = 0 := by
    rcases mul_eq_zero.mp hsu with h | h
    · exact h
    · exact absurd (sub_eq_zero.mp h) huv
  rw [ht, hs0] at h2
  simp at h2
  exact hη h2

theorem vec3_injective {u v w : E} (huv : u ≠ v) (huw : u ≠ w) (hvw : v ≠ w) :
    Function.Injective ![u, v, w] := by
  intro i j hij
  fin_cases i <;> fin_cases j <;> simp_all

/-- The Vandermonde map `(s,t,r) ↦ s(1,u,u²) + t(1,v,v²) + r(1,w,w²)`, `E`-linear hence `K`-linear. -/
def vanderMap (u v w : E) : (Fin 3 → E) →ₗ[K] (Fin 3 → E) :=
  (Matrix.vecMulLinear (Matrix.vandermonde ![u, v, w])).restrictScalars K

theorem vanderMap_apply (u v w : E) (c : Fin 3 → E) (i : Fin 3) :
    vanderMap K u v w c i = c 0 * u ^ (i : ℕ) + c 1 * v ^ (i : ℕ) + c 2 * w ^ (i : ℕ) := by
  simp [vanderMap, Matrix.vecMul, dotProduct, Fin.sum_univ_three, Matrix.vandermonde_apply]

theorem vanderMap_eq_sum (u v w : E) (c : Fin 3 → E) :
    vanderMap K u v w c = conicMap K u (c 0) + conicMap K v (c 1) + conicMap K w (c 2) := by
  funext i
  simp [vanderMap_apply, conicMap_apply]

theorem vanderMap_bijective {u v w : E} (huv : u ≠ v) (huw : u ≠ w) (hvw : v ≠ w) :
    Function.Bijective (vanderMap K u v w) := by
  have hunit : IsUnit (Matrix.vandermonde ![u, v, w]) := by
    rw [Matrix.isUnit_iff_isUnit_det, isUnit_iff_ne_zero]
    exact Matrix.det_vandermonde_ne_zero_iff.mpr (vec3_injective huv huw hvw)
  have h1 : Function.Injective (vanderMap K u v w) :=
    Matrix.vecMul_injective_iff_isUnit.mpr hunit
  have h2 : Function.Surjective (vanderMap K u v w) :=
    Matrix.vecMul_surjective_iff_isUnit.mpr hunit
  exact ⟨h1, h2⟩

/-- Three distinct conic lines span `U`. -/
theorem conicLine_sup_three_eq_top {u v w : E} (huv : u ≠ v) (huw : u ≠ w) (hvw : v ≠ w) :
    conicLine K u ⊔ conicLine K v ⊔ conicLine K w = ⊤ := by
  rw [eq_top_iff]
  intro x _
  obtain ⟨c, rfl⟩ := (vanderMap_bijective K huv huw hvw).surjective x
  rw [vanderMap_eq_sum]
  refine add_mem (add_mem ?_ ?_) ?_
  · exact mem_sup_left (mem_sup_left (LinearMap.mem_range.mpr ⟨c 0, rfl⟩))
  · exact mem_sup_left (mem_sup_right (LinearMap.mem_range.mpr ⟨c 1, rfl⟩))
  · exact mem_sup_right (LinearMap.mem_range.mpr ⟨c 2, rfl⟩)

/-- Three distinct conic lines are independent: a vanishing sum has vanishing parts. -/
theorem conicMap_sum_eq_zero {u v w : E} (huv : u ≠ v) (huw : u ≠ w) (hvw : v ≠ w)
    {s t r : E} (h : conicMap K u s + conicMap K v t + conicMap K w r = 0) :
    s = 0 ∧ t = 0 ∧ r = 0 := by
  have h' : vanderMap K u v w ![s, t, r] = vanderMap K u v w 0 := by
    rw [vanderMap_eq_sum, map_zero]
    simpa using h
  have := (vanderMap_bijective K huv huw hvw).injective h'
  refine ⟨?_, ?_, ?_⟩
  · simpa using congrFun this 0
  · simpa using congrFun this 1
  · simpa using congrFun this 2

/-- `dim_K U = 3m`. -/
theorem finrank_U [FiniteDimensional K E] :
    Module.finrank K (Fin 3 → E) = 3 * Module.finrank K E := by
  rw [Module.finrank_pi_fintype]
  simp

end Conic

/-! ### Dimension bookkeeping for images with trivial kernel intersection -/

section FinrankMap

variable {K V W : Type*} [Field K] [AddCommGroup V] [Module K V] [AddCommGroup W] [Module K W]

/-- `dim f(A) = dim A` when `A ∩ ker f = 0`. -/
theorem finrank_map_of_disjoint_ker [FiniteDimensional K V] (f : V →ₗ[K] W)
    (A : Submodule K V) (h : Disjoint A (LinearMap.ker f)) :
    Module.finrank K (A.map f) = Module.finrank K A := by
  have h1 : A.map f = LinearMap.range (f ∘ₗ A.subtype) := by
    rw [LinearMap.range_comp, Submodule.range_subtype]
  have h2 : LinearMap.ker (f ∘ₗ A.subtype) = ⊥ := by
    rw [LinearMap.ker_comp, ← Submodule.disjoint_iff_comap_eq_bot]
    exact h
  rw [h1, LinearMap.finrank_range_of_inj (LinearMap.ker_eq_bot.mp h2)]

end FinrankMap

/-! ### The projected family `R_u = π(L_u)` in `C = U / ⟨h⟩` -/

section Projected

variable (K : Type*) {E : Type*} [Field K] [Field E] [Algebra K E]

/-- The line `⟨h⟩`. -/
abbrev hLine (η : E) : Submodule K (Fin 3 → E) := K ∙ hvec η

/-- `C = U / ⟨h⟩`, of dimension `3m - 1`. -/
abbrev ConicQuot (η : E) := (Fin 3 → E) ⧸ hLine K η

/-- `R_u = π(L_u)` of (2). -/
def projLine (η u : E) : Submodule K (ConicQuot K η) :=
  (conicLine K u).map (hLine K η).mkQ

theorem disjoint_conicLine_hLine {η u : E} (hη : η ≠ 0) :
    Disjoint (conicLine K u) (hLine K η) :=
  (Submodule.disjoint_span_singleton' (hvec_ne_zero hη)).mpr (hvec_not_mem_conicLine K hη)

theorem disjoint_conicLine_sup_hLine {η u v : E} (huv : u ≠ v) (hη : η ≠ 0) :
    Disjoint (conicLine K u ⊔ conicLine K v) (hLine K η) :=
  (Submodule.disjoint_span_singleton' (hvec_ne_zero hη)).mpr (hvec_not_mem_sup K huv hη)

variable [FiniteDimensional K E]

/-- `dim C + 1 = 3m`, i.e. `dim C = 3m - 1`. -/
theorem finrank_conicQuot {η : E} (hη : η ≠ 0) :
    Module.finrank K (ConicQuot K η) + 1 = 3 * Module.finrank K E := by
  have := Submodule.finrank_quotient_add_finrank (hLine K η)
  rw [finrank_span_singleton (hvec_ne_zero hη), finrank_U] at this
  exact this

/-- (3): `dim R_u = m`. -/
theorem finrank_projLine {η : E} (hη : η ≠ 0) (u : E) :
    Module.finrank K (projLine K η u) = Module.finrank K E := by
  unfold projLine
  rw [finrank_map_of_disjoint_ker _ _ ?_, finrank_conicLine]
  rw [Submodule.ker_mkQ]
  exact disjoint_conicLine_hLine K hη

/-- (3): `R_u ∩ R_v = 0` for `u ≠ v`. -/
theorem disjoint_projLine {η u v : E} (huv : u ≠ v) (hη : η ≠ 0) :
    Disjoint (projLine K η u) (projLine K η v) := by
  unfold projLine
  refine disjoint_map_of_disjoint _ (disjoint_conicLine K huv) ?_
  rw [Submodule.ker_mkQ]
  exact disjoint_conicLine_sup_hLine K huv hη

/-- (3): `dim (R_u + R_v) = 2m`. -/
theorem finrank_projLine_sup {η u v : E} (huv : u ≠ v) (hη : η ≠ 0) :
    Module.finrank K ↥(projLine K η u ⊔ projLine K η v) = 2 * Module.finrank K E := by
  unfold projLine
  rw [← Submodule.map_sup, finrank_map_of_disjoint_ker _ _ ?_]
  · have h := Submodule.finrank_sup_add_finrank_inf_eq (conicLine K u) (conicLine K v)
    rw [disjoint_iff.mp (disjoint_conicLine K huv), finrank_bot, add_zero,
      finrank_conicLine, finrank_conicLine] at h
    rw [h]; ring
  · rw [Submodule.ker_mkQ]
    exact disjoint_conicLine_sup_hLine K huv hη

/-- (4): `R_u + R_v + R_w = C`. -/
theorem projLine_sup_three_eq_top {η u v w : E} (huv : u ≠ v) (huw : u ≠ w) (hvw : v ≠ w) :
    projLine K η u ⊔ projLine K η v ⊔ projLine K η w = ⊤ := by
  unfold projLine
  rw [← Submodule.map_sup, ← Submodule.map_sup, conicLine_sup_three_eq_top K huv huw hvw,
    Submodule.map_top, Submodule.range_mkQ]

/-- (4): the kernel of `R_u ⊕ R_v ⊕ R_w → C` is one-dimensional, in the form
`dim R_u + dim R_v + dim R_w = dim C + 1`. -/
theorem finrank_projLine_three {η u v w : E} (hη : η ≠ 0) :
    Module.finrank K (projLine K η u) + Module.finrank K (projLine K η v) +
      Module.finrank K (projLine K η w) = Module.finrank K (ConicQuot K η) + 1 := by
  rw [finrank_projLine K hη, finrank_projLine K hη, finrank_projLine K hη, finrank_conicQuot K hη]
  ring

/-- The complete observer kernel `K_u = R_u^⊥ ≤ C*`. -/
def obsKernel (η u : E) : Submodule K (Module.Dual K (ConicQuot K η)) :=
  (projLine K η u).dualAnnihilator

/-- (5): `dim K_u = 2m - 1`, in the form `dim K_u + m = 3m - 1`. -/
theorem finrank_obsKernel {η : E} (hη : η ≠ 0) (u : E) :
    Module.finrank K (obsKernel K η u) + Module.finrank K E = Module.finrank K (ConicQuot K η) := by
  rw [← finrank_projLine K hη u, add_comm]
  exact Subspace.finrank_add_finrank_dualAnnihilator_eq _

/-- (5): `dim (K_u ∩ K_v) = m - 1`, in the form `dim (K_u ∩ K_v) + 2m = 3m - 1`. -/
theorem finrank_obsKernel_inf {η u v : E} (huv : u ≠ v) (hη : η ≠ 0) :
    Module.finrank K ↥(obsKernel K η u ⊓ obsKernel K η v) + 2 * Module.finrank K E =
      Module.finrank K (ConicQuot K η) := by
  unfold obsKernel
  rw [← Submodule.dualAnnihilator_sup_eq, ← finrank_projLine_sup K huv hη, add_comm]
  exact Subspace.finrank_add_finrank_dualAnnihilator_eq _

/-- (5): `K_u + K_v = C*` (two complete anchors). -/
theorem obsKernel_sup_eq_top {η u v : E} (huv : u ≠ v) (hη : η ≠ 0) :
    obsKernel K η u ⊔ obsKernel K η v = ⊤ := by
  unfold obsKernel
  rw [← Subspace.dualAnnihilator_inf_eq, disjoint_iff.mp (disjoint_projLine K huv hη),
    Submodule.dualAnnihilator_bot]

end Projected

/-! ### Codimension-one unprojection (Theorem target D) -/

section Unprojection

variable {K C : Type*} [Field K] [AddCommGroup C] [Module K C]

/-- The lift of `R ≤ C` along `lam : R → K`: the graph `{(r, lam r)} ≤ C × K`. -/
def liftSub (R : Submodule K C) (lam : Module.Dual K R) : Submodule K (C × K) :=
  LinearMap.range (R.subtype.prod lam)

theorem mem_liftSub {R : Submodule K C} {lam : Module.Dual K R} {x : C × K} :
    x ∈ liftSub R lam ↔ ∃ r : R, ((r : C), lam r) = x := by
  simp [liftSub, LinearMap.mem_range]

/-- The lift projects back onto `R`. -/
theorem liftSub_map_fst (R : Submodule K C) (lam : Module.Dual K R) :
    (liftSub R lam).map (LinearMap.fst K C K) = R := by
  unfold liftSub
  rw [← LinearMap.range_comp, LinearMap.fst_prod, Submodule.range_subtype]

/-- The lift meets `0 × K` trivially (it is a graph). -/
theorem disjoint_liftSub_ker_fst (R : Submodule K C) (lam : Module.Dual K R) :
    Disjoint (liftSub R lam) (LinearMap.ker (LinearMap.fst K C K)) := by
  rw [Submodule.disjoint_def]
  rintro _ ⟨x, rfl⟩ hx
  simp only [LinearMap.mem_ker, LinearMap.prod_apply, LinearMap.fst_apply, Submodule.coe_subtype] at hx
  have : x = 0 := Subtype.ext hx
  simp [this]

theorem liftSub_injective (R : Submodule K C) (lam : Module.Dual K R) :
    Function.Injective (R.subtype.prod lam) := fun _ _ h =>
  Subtype.ext (congrArg Prod.fst h)

/-- `dim R̃ = dim R`. -/
theorem finrank_liftSub [FiniteDimensional K C] (R : Submodule K C) (lam : Module.Dual K R) :
    Module.finrank K (liftSub R lam) = Module.finrank K R :=
  LinearMap.finrank_range_of_inj (liftSub_injective R lam)

/-- **Graphs are lifts.**  A subspace `G ≤ C × K` meeting `0 × K` trivially is the lift
of its projection along a unique linear functional (Mathlib's `Submodule.toLinearPMap`). -/
theorem exists_liftSub_eq (G : Submodule K (C × K))
    (hG : Disjoint G (LinearMap.ker (LinearMap.fst K C K))) :
    ∃ lam : Module.Dual K (G.map (LinearMap.fst K C K)),
      liftSub (G.map (LinearMap.fst K C K)) lam = G := by
  have hg : ∀ x ∈ G, x.1 = 0 → x.2 = 0 := by
    intro x hx h1
    have : x = 0 := (Submodule.disjoint_def.mp hG) x hx (by simpa [LinearMap.mem_ker] using h1)
    rw [this]; rfl
  refine ⟨G.toLinearPMap.toFun, ?_⟩
  show liftSub G.toLinearPMap.domain G.toLinearPMap.toFun = G
  calc liftSub G.toLinearPMap.domain G.toLinearPMap.toFun = G.toLinearPMap.graph := by
        ext ⟨c, t⟩
        rw [mem_liftSub, LinearPMap.mem_graph_iff]
        simp
    _ = G := G.toLinearPMap_graph_eq hg

/-- The sum map of a triple, `(x, y, z) ↦ x + y + z`. -/
def tripleSum (R₁ R₂ R₃ : Submodule K C) : R₁ × R₂ × R₃ →ₗ[K] C :=
  R₁.subtype.coprod (R₂.subtype.coprod R₃.subtype)

/-- The lift functional of a triple, `(x, y, z) ↦ λ₁ x + λ₂ y + λ₃ z`. -/
def tripleLam {R₁ R₂ R₃ : Submodule K C} (l₁ : Module.Dual K R₁) (l₂ : Module.Dual K R₂)
    (l₃ : Module.Dual K R₃) : R₁ × R₂ × R₃ →ₗ[K] K :=
  l₁.coprod (l₂.coprod l₃)

/-- The lifted sum map of a triple. -/
def tripleLift {R₁ R₂ R₃ : Submodule K C} (l₁ : Module.Dual K R₁) (l₂ : Module.Dual K R₂)
    (l₃ : Module.Dual K R₃) : R₁ × R₂ × R₃ →ₗ[K] C × K :=
  (tripleSum R₁ R₂ R₃).prod (tripleLam l₁ l₂ l₃)

theorem tripleSum_apply {R₁ R₂ R₃ : Submodule K C} (x : R₁ × R₂ × R₃) :
    tripleSum R₁ R₂ R₃ x = (x.1 : C) + ((x.2.1 : C) + (x.2.2 : C)) := rfl

theorem tripleLam_apply {R₁ R₂ R₃ : Submodule K C} (l₁ : Module.Dual K R₁)
    (l₂ : Module.Dual K R₂) (l₃ : Module.Dual K R₃) (x : R₁ × R₂ × R₃) :
    tripleLam l₁ l₂ l₃ x = l₁ x.1 + (l₂ x.2.1 + l₃ x.2.2) := rfl

/-- The range of the lifted sum map is the sum of the three lifts. -/
theorem range_tripleLift {R₁ R₂ R₃ : Submodule K C} (l₁ : Module.Dual K R₁)
    (l₂ : Module.Dual K R₂) (l₃ : Module.Dual K R₃) :
    LinearMap.range (tripleLift l₁ l₂ l₃) = liftSub R₁ l₁ ⊔ liftSub R₂ l₂ ⊔ liftSub R₃ l₃ := by
  apply le_antisymm
  · rintro _ ⟨⟨x, y, z⟩, rfl⟩
    have : tripleLift l₁ l₂ l₃ (x, y, z) =
        (R₁.subtype.prod l₁) x + (R₂.subtype.prod l₂) y + (R₃.subtype.prod l₃) z := by
      ext <;> simp [tripleLift, tripleSum_apply, tripleLam_apply, add_assoc]
    rw [this]
    refine add_mem (add_mem ?_ ?_) ?_
    · exact mem_sup_left (mem_sup_left (LinearMap.mem_range_self _ _))
    · exact mem_sup_left (mem_sup_right (LinearMap.mem_range_self _ _))
    · exact mem_sup_right (LinearMap.mem_range_self _ _)
  · refine sup_le (sup_le ?_ ?_) ?_ <;> rintro _ ⟨x, rfl⟩
    · exact ⟨(x, 0, 0), by ext <;> simp [tripleLift, tripleSum_apply, tripleLam_apply]⟩
    · exact ⟨(0, x, 0), by ext <;> simp [tripleLift, tripleSum_apply, tripleLam_apply]⟩
    · exact ⟨(0, 0, x), by ext <;> simp [tripleLift, tripleSum_apply, tripleLam_apply]⟩

/-- **The lift equation.**  If `ker Φ = K·a` with `a ≠ 0`, then `Φ.prod ψ` is injective iff
`ψ a ≠ 0`.  Over `GF(2)` this is the linear equation `ψ a = 1`. -/
theorem ker_prod_eq_bot_iff {M N : Type*} [AddCommGroup M] [Module K M] [AddCommGroup N]
    [Module K N] (Φ : M →ₗ[K] N) (ψ : M →ₗ[K] K) {a : M} (ha : a ≠ 0)
    (hker : LinearMap.ker Φ = K ∙ a) :
    LinearMap.ker (Φ.prod ψ) = ⊥ ↔ ψ a ≠ 0 := by
  rw [LinearMap.ker_prod, hker]
  constructor
  · intro h hψ
    have : a ∈ (K ∙ a) ⊓ LinearMap.ker ψ := ⟨Submodule.mem_span_singleton_self a, hψ⟩
    rw [h] at this
    exact ha ((Submodule.mem_bot K).mp this)
  · intro hψ
    rw [eq_bot_iff]
    rintro x ⟨hx1, hx2⟩
    obtain ⟨c, rfl⟩ := Submodule.mem_span_singleton.mp hx1
    have h2 : ψ (c • a) = 0 := hx2
    rw [map_smul, smul_eq_mul] at h2
    rcases mul_eq_zero.mp h2 with hc | hc
    · simp [hc]
    · exact absurd hc hψ

/-- **Lift equation for a triple**: with the relation space `ker (R₁ ⊕ R₂ ⊕ R₃ → C) = K·a`,
the lifted triple is direct iff `λ₁ a₁ + λ₂ a₂ + λ₃ a₃ ≠ 0`. -/
theorem tripleLift_ker_eq_bot_iff {R₁ R₂ R₃ : Submodule K C} (l₁ : Module.Dual K R₁)
    (l₂ : Module.Dual K R₂) (l₃ : Module.Dual K R₃) {a : R₁ × R₂ × R₃} (ha : a ≠ 0)
    (hker : LinearMap.ker (tripleSum R₁ R₂ R₃) = K ∙ a) :
    LinearMap.ker (tripleLift l₁ l₂ l₃) = ⊥ ↔ tripleLam l₁ l₂ l₃ a ≠ 0 :=
  ker_prod_eq_bot_iff _ _ ha hker

/-- After an admitted lift the lifted triple has full rank `dim R₁ + dim R₂ + dim R₃`. -/
theorem finrank_lifted_triple [FiniteDimensional K C] {R₁ R₂ R₃ : Submodule K C}
    (l₁ : Module.Dual K R₁) (l₂ : Module.Dual K R₂) (l₃ : Module.Dual K R₃)
    (h : LinearMap.ker (tripleLift l₁ l₂ l₃) = ⊥) :
    Module.finrank K ↥(liftSub R₁ l₁ ⊔ liftSub R₂ l₂ ⊔ liftSub R₃ l₃) =
      Module.finrank K R₁ + Module.finrank K R₂ + Module.finrank K R₃ := by
  rw [← range_tripleLift, LinearMap.finrank_range_of_inj (LinearMap.ker_eq_bot.mp h),
    Module.finrank_prod, Module.finrank_prod, add_assoc]

/-- The gauge lift of `ℓ ∈ C*` on `R`: `λ = ℓ|_R`. -/
def gaugeLam (R : Submodule K C) (ℓ : Module.Dual K C) : Module.Dual K R := ℓ ∘ₗ R.subtype

theorem tripleLam_gauge {R₁ R₂ R₃ : Submodule K C} (ℓ : Module.Dual K C) :
    tripleLam (gaugeLam R₁ ℓ) (gaugeLam R₂ ℓ) (gaugeLam R₃ ℓ) = ℓ ∘ₗ tripleSum R₁ R₂ R₃ := by
  ext x <;> simp [tripleLam_apply, tripleSum_apply, gaugeLam]

/-- **Gauge lifts are homogeneous solutions**: they vanish on every relation. -/
theorem tripleLam_gauge_eq_zero {R₁ R₂ R₃ : Submodule K C} (ℓ : Module.Dual K C)
    {a : R₁ × R₂ × R₃} (ha : a ∈ LinearMap.ker (tripleSum R₁ R₂ R₃)) :
    tripleLam (gaugeLam R₁ ℓ) (gaugeLam R₂ ℓ) (gaugeLam R₃ ℓ) a = 0 := by
  rw [tripleLam_gauge, LinearMap.comp_apply, LinearMap.mem_ker.mp ha, map_zero]

theorem tripleLam_add {R₁ R₂ R₃ : Submodule K C} (l₁ g₁ : Module.Dual K R₁)
    (l₂ g₂ : Module.Dual K R₂) (l₃ g₃ : Module.Dual K R₃) (a : R₁ × R₂ × R₃) :
    tripleLam (l₁ + g₁) (l₂ + g₂) (l₃ + g₃) a = tripleLam l₁ l₂ l₃ a + tripleLam g₁ g₂ g₃ a := by
  simp only [tripleLam_apply, LinearMap.add_apply]
  ring

/-- **The gauge action preserves the lift equation.** -/
theorem tripleLam_add_gauge {R₁ R₂ R₃ : Submodule K C} (l₁ : Module.Dual K R₁)
    (l₂ : Module.Dual K R₂) (l₃ : Module.Dual K R₃) (ℓ : Module.Dual K C)
    {a : R₁ × R₂ × R₃} (ha : a ∈ LinearMap.ker (tripleSum R₁ R₂ R₃)) :
    tripleLam (l₁ + gaugeLam R₁ ℓ) (l₂ + gaugeLam R₂ ℓ) (l₃ + gaugeLam R₃ ℓ) a =
      tripleLam l₁ l₂ l₃ a := by
  rw [tripleLam_add, tripleLam_gauge_eq_zero ℓ ha, add_zero]

/-- The shear `(c, t) ↦ (c, t + ℓ c)` of `C × K`. -/
def shear (ℓ : Module.Dual K C) : C × K →ₗ[K] C × K :=
  (LinearMap.fst K C K).prod (LinearMap.snd K C K + ℓ ∘ₗ LinearMap.fst K C K)

/-- **The gauge action is the shear**: `lift (λ + ℓ|_R) = shear ℓ (lift λ)`. -/
theorem liftSub_add_gauge (R : Submodule K C) (lam : Module.Dual K R) (ℓ : Module.Dual K C) :
    liftSub R (lam + gaugeLam R ℓ) = (liftSub R lam).map (shear ℓ) := by
  unfold liftSub
  rw [← LinearMap.range_comp]
  have : shear ℓ ∘ₗ R.subtype.prod lam = R.subtype.prod (lam + gaugeLam R ℓ) := by
    ext x <;> simp [shear, gaugeLam]
  rw [this]

end Unprojection

/-! ### Existence of the lift for a family split by a line -/

section LiftExistence

variable {K U : Type*} [Field K] [AddCommGroup U] [Module K U]

/-- A functional taking the value `1` on a nonzero vector. -/
theorem exists_dual_apply_eq_one {h : U} (hh : h ≠ 0) : ∃ ξ : Module.Dual K U, ξ h = 1 := by
  obtain ⟨φ, hφ⟩ := not_forall.mp ((Module.forall_dual_apply_eq_zero_iff K h).not.mpr hh)
  exact ⟨(φ h)⁻¹ • φ, by simp [inv_mul_cancel₀ hφ]⟩

/-- The identification `θ = (π, ξ) : U → (U/⟨h⟩) × K`. -/
def splitMap (h : U) (ξ : Module.Dual K U) : U →ₗ[K] (U ⧸ (K ∙ h)) × K :=
  (K ∙ h).mkQ.prod ξ

theorem splitMap_bijective {h : U} (ξ : Module.Dual K U) (hξ : ξ h = 1) :
    Function.Bijective (splitMap h ξ) := by
  constructor
  · rw [← LinearMap.ker_eq_bot, splitMap, LinearMap.ker_prod, Submodule.ker_mkQ, eq_bot_iff]
    rintro x ⟨hx1, hx2⟩
    obtain ⟨c, rfl⟩ := Submodule.mem_span_singleton.mp hx1
    have h2 : ξ (c • h) = 0 := hx2
    rw [map_smul, hξ, smul_eq_mul, mul_one] at h2
    simp [h2]
  · rintro ⟨q, t⟩
    obtain ⟨x, rfl⟩ := (K ∙ h).mkQ_surjective q
    refine ⟨x + (t - ξ x) • h, ?_⟩
    have hh : (K ∙ h).mkQ h = 0 := by
      rw [Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero]
      exact Submodule.mem_span_singleton_self h
    ext
    · simp [splitMap, hh]
    · simp [splitMap, hξ]

/-- **Existence of the lift.**  Let `L_j ≤ U` be subspaces with `L_j ∩ ⟨h⟩ = 0`.  Put
`R_j = π(L_j) ≤ C = U/⟨h⟩`.  Then each `θ(L_j)` is the lift of `R_j` along some
`λ_j : R_j → K`, and for every triple with `L_i + L_j + L_l = U` and
`dim L_i + dim L_j + dim L_l = dim U` the lifted triple is a direct decomposition of
`C × K`: its sum map is injective with range `⊤`. -/
theorem exists_lift [FiniteDimensional K U] {h : U} (hh : h ≠ 0) {ι : Type*}
    (L : ι → Submodule K U) (hL : ∀ j, Disjoint (L j) (K ∙ h)) :
    ∃ (ξ : Module.Dual K U) (lam : ∀ j, Module.Dual K
        (((L j).map (splitMap h ξ)).map (LinearMap.fst K (U ⧸ (K ∙ h)) K))),
      ξ h = 1 ∧
      (∀ j, ((L j).map (splitMap h ξ)).map (LinearMap.fst K (U ⧸ (K ∙ h)) K) =
        (L j).map (K ∙ h).mkQ) ∧
      (∀ j, liftSub _ (lam j) = (L j).map (splitMap h ξ)) ∧
      (∀ i j l, L i ⊔ L j ⊔ L l = ⊤ →
        Module.finrank K (L i) + Module.finrank K (L j) + Module.finrank K (L l) =
          Module.finrank K U →
        LinearMap.ker (tripleLift (lam i) (lam j) (lam l)) = ⊥ ∧
        LinearMap.range (tripleLift (lam i) (lam j) (lam l)) = ⊤) := by
  obtain ⟨ξ, hξ⟩ := exists_dual_apply_eq_one (K := K) hh
  have hproj : ∀ j, ((L j).map (splitMap h ξ)).map (LinearMap.fst K (U ⧸ (K ∙ h)) K) =
      (L j).map (K ∙ h).mkQ := by
    intro j
    rw [← Submodule.map_comp]
    rfl
  have hgraph : ∀ j, Disjoint ((L j).map (splitMap h ξ))
      (LinearMap.ker (LinearMap.fst K (U ⧸ (K ∙ h)) K)) := by
    intro j
    rw [Submodule.disjoint_def]
    rintro _ ⟨x, hx, rfl⟩ hker
    have hx0 : (K ∙ h).mkQ x = 0 := by simpa [splitMap] using hker
    rw [Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero] at hx0
    have : x = 0 := (Submodule.disjoint_def.mp (hL j)) x hx hx0
    simp [this]
  choose lam hlam using fun j => exists_liftSub_eq _ (hgraph j)
  refine ⟨ξ, lam, hξ, hproj, hlam, ?_⟩
  intro i j l hsup hdim
  have hθ := splitMap_bijective ξ hξ
  have hrange : LinearMap.range (tripleLift (lam i) (lam j) (lam l)) = ⊤ := by
    rw [range_tripleLift, hlam, hlam, hlam, ← Submodule.map_sup, ← Submodule.map_sup, hsup,
      Submodule.map_top, LinearMap.range_eq_top.mpr hθ.surjective]
  have hdim' : Module.finrank K (((L i).map (splitMap h ξ)).map (LinearMap.fst K _ K) ×
      ((L j).map (splitMap h ξ)).map (LinearMap.fst K _ K) ×
      ((L l).map (splitMap h ξ)).map (LinearMap.fst K _ K)) =
      Module.finrank K ((U ⧸ (K ∙ h)) × K) := by
    have hfin : ∀ j, Module.finrank K (((L j).map (splitMap h ξ)).map (LinearMap.fst K _ K)) =
        Module.finrank K (L j) := by
      intro j
      rw [hproj, finrank_map_of_disjoint_ker _ _ (by rw [Submodule.ker_mkQ]; exact hL j)]
    rw [Module.finrank_prod, Module.finrank_prod, hfin, hfin, hfin, ← add_assoc, hdim,
      LinearEquiv.finrank_eq (LinearEquiv.ofBijective _ hθ)]
  exact ⟨(LinearMap.ker_eq_bot_iff_range_eq_top_of_finrank_eq_finrank hdim').mpr hrange, hrange⟩

end LiftExistence

/-! ### The projected-conic family admits a lift (Theorem target D, existence) -/

section ConicLift

variable (K : Type*) {E : Type*} [Field K] [Field E] [Algebra K E] [FiniteDimensional K E]

/-- **Existence for the genuine projected-conic family.**  With `L_u = E·(1,u,u²)` and
`h = (0,0,η)`, `η ≠ 0`: there is a lift of every `R_u = π(L_u)` such that every triple
of distinct labels lifts to a direct decomposition of `C × K`. -/
theorem conic_exists_lift {η : E} (hη : η ≠ 0) :
    ∃ (ξ : Module.Dual K (Fin 3 → E)) (lam : ∀ u : E, Module.Dual K
        (((conicLine K u).map (splitMap (hvec η) ξ)).map (LinearMap.fst K _ K))),
      ξ (hvec η) = 1 ∧
      (∀ u, ((conicLine K u).map (splitMap (hvec η) ξ)).map (LinearMap.fst K _ K) =
        projLine K η u) ∧
      (∀ u, liftSub _ (lam u) = (conicLine K u).map (splitMap (hvec η) ξ)) ∧
      (∀ u v w : E, u ≠ v → u ≠ w → v ≠ w →
        LinearMap.ker (tripleLift (lam u) (lam v) (lam w)) = ⊥ ∧
        LinearMap.range (tripleLift (lam u) (lam v) (lam w)) = ⊤) := by
  obtain ⟨ξ, lam, hξ, hproj, hlam, htriple⟩ :=
    exists_lift (K := K) (hvec_ne_zero hη) (conicLine K)
      (fun u => disjoint_conicLine_hLine K hη)
  refine ⟨ξ, lam, hξ, hproj, hlam, ?_⟩
  intro u v w huv huw hvw
  refine htriple u v w (conicLine_sup_three_eq_top K huv huw hvw) ?_
  rw [finrank_conicLine, finrank_conicLine, finrank_conicLine, finrank_U]
  ring

end ConicLift

end TwoAnchorPairCore
