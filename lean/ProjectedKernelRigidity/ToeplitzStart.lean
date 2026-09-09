import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.LinearAlgebra.Dimension.Finite
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Finsupp.LinearCombination
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Finset.Max
import Mathlib.Logic.Equiv.Sum
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Probability.ConditionalProbability
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.MeasureTheory.Integral.Lebesgue.Add
import Mathlib.Data.ZMod.Basic
import Mathlib.LinearAlgebra.Matrix.Notation

/-!
# Toeplitz right-start visibility over finite fields

This module formalizes the theorems of `toeplitz_right_start_problem.md`
(Theorems A, B, C, D and the finite counting statement behind B) for the
frozen convention

  `Y(σ)_{i,j} = σ_{i-j+b-1}`,   `0 ≤ i < N`, `0 ≤ j < b`,

where the seed `σ` has `N + b - 1` coordinates.

The module is independent of the projected-kernel definitions of the rest of
the project.  Every statement is over an arbitrary field; the counting and
probability statements additionally assume the field is finite.

Probabilities are stated as exact cardinalities of seed sets: with `q = |F|`,
the uniform measure on the `q^(N+b-1)` seeds assigns probability
`#S / q^(N+b-1)` to a seed set `S`.  The ratio forms are also stated in `ℚ`.

No independence between the events `(λψ)Y = 0` for different `λ` is used:
Theorem B is a union bound (`Finset.card_biUnion_le`).
-/

set_option linter.unusedSectionVars false

namespace ToeplitzStart

open Matrix Finset

variable {F : Type*} [Field F]

/-! ## The frozen convention -/

/-- Convention (2): `toeplitz N b σ i j = σ (i - j + b - 1)`.  The index is
written with natural-number arithmetic as `i + (b - 1 - j)`, which is the same
integer because `j < b`. -/
def toeplitz (N b : ℕ) (σ : Fin (N + b - 1) → F) : Matrix (Fin N) (Fin b) F :=
  fun i j => σ ⟨i.val + (b - 1 - j.val), by have := i.isLt; have := j.isLt; omega⟩

theorem toeplitz_apply (N b : ℕ) (σ : Fin (N + b - 1) → F) (i : Fin N) (j : Fin b) :
    toeplitz N b σ i j =
      σ ⟨i.val + (b - 1 - j.val), by have := i.isLt; have := j.isLt; omega⟩ := rfl

/-- The seed index of entry `(i, j)` is `i - j + b - 1` as an integer. -/
theorem toeplitz_index (N b : ℕ) (i : Fin N) (j : Fin b) :
    ((i.val + (b - 1 - j.val) : ℕ) : ℤ) = (i.val : ℤ) - j.val + b - 1 := by
  have := j.isLt
  omega

/-- The matrix is constant along diagonals: entries `(i, j)` and `(i + 1, j + 1)`
agree. -/
theorem toeplitz_diagonal (N b : ℕ) (σ : Fin (N + b - 1) → F) (i : Fin N) (j : Fin b)
    (hi : i.val + 1 < N) (hj : j.val + 1 < b) :
    toeplitz N b σ ⟨i.val + 1, hi⟩ ⟨j.val + 1, hj⟩ = toeplitz N b σ i j := by
  simp only [toeplitz_apply]
  congr 1
  ext
  simp only
  omega

theorem toeplitz_add {N b : ℕ} (σ τ : Fin (N + b - 1) → F) :
    toeplitz N b (σ + τ) = toeplitz N b σ + toeplitz N b τ := by
  ext i j; simp [toeplitz]

theorem toeplitz_smul {N b : ℕ} (c : F) (σ : Fin (N + b - 1) → F) :
    toeplitz N b (c • σ) = c • toeplitz N b σ := by
  ext i j; simp [toeplitz]

/-- The fixed-row map `T_a : σ ↦ a Y(σ)` as a linear map. -/
def rowMap (N b : ℕ) (a : Fin N → F) : (Fin (N + b - 1) → F) →ₗ[F] (Fin b → F) where
  toFun σ := a ᵥ* toeplitz N b σ
  map_add' σ τ := by rw [toeplitz_add, Matrix.vecMul_add]
  map_smul' c σ := by rw [toeplitz_smul, Matrix.vecMul_smul]; rfl

@[simp] theorem rowMap_apply (N b : ℕ) (a : Fin N → F) (σ : Fin (N + b - 1) → F) :
    rowMap N b a σ = a ᵥ* toeplitz N b σ := rfl

/-! ## Theorem A: fixed-row surjectivity -/

/-- Zero-extension of a width-`b` vector onto the seed window `[i0, i0 + b)`. -/
def window (N b : ℕ) (i0 : ℕ) (τ : Fin b → F) : Fin (N + b - 1) → F :=
  fun m => if h : i0 ≤ m.val ∧ m.val < i0 + b then τ ⟨m.val - i0, by omega⟩ else 0

theorem window_apply_of_mem {N b : ℕ} (i0 : ℕ) (τ : Fin b → F) (m : Fin (N + b - 1))
    (h : i0 ≤ m.val ∧ m.val < i0 + b) :
    window N b i0 τ m = τ ⟨m.val - i0, by omega⟩ := by
  simp [window, h]

theorem window_apply_of_not_mem {N b : ℕ} (i0 : ℕ) (τ : Fin b → F) (m : Fin (N + b - 1))
    (h : ¬ (i0 ≤ m.val ∧ m.val < i0 + b)) :
    window N b i0 τ m = 0 := by
  simp [window, h]

theorem window_add {N b : ℕ} (i0 : ℕ) (τ υ : Fin b → F) :
    window N b i0 (τ + υ) = window N b i0 τ + window N b i0 υ := by
  ext m; simp only [window, Pi.add_apply]; split_ifs <;> simp

theorem window_smul {N b : ℕ} (i0 : ℕ) (c : F) (τ : Fin b → F) :
    window N b i0 (c • τ) = c • window N b i0 τ := by
  ext m; simp only [window, Pi.smul_apply]; split_ifs <;> simp

/-- The window embedding as a linear map. -/
def windowMap (N b : ℕ) (i0 : ℕ) : (Fin b → F) →ₗ[F] (Fin (N + b - 1) → F) where
  toFun := window N b i0
  map_add' := window_add i0
  map_smul' c τ := by rw [window_smul]; rfl

@[simp] theorem windowMap_apply (N b : ℕ) (i0 : ℕ) (τ : Fin b → F) :
    windowMap N b i0 τ = window N b i0 τ := rfl

/-- A nonzero vector has a last nonzero coordinate. -/
theorem exists_last_nonzero {N : ℕ} (a : Fin N → F) (ha : a ≠ 0) :
    ∃ i0 : Fin N, a i0 ≠ 0 ∧ ∀ i, i0 < i → a i = 0 := by
  classical
  have hne : (univ.filter fun i : Fin N => a i ≠ 0).Nonempty := by
    by_contra h
    apply ha
    ext i
    rw [Finset.not_nonempty_iff_eq_empty, Finset.filter_eq_empty_iff] at h
    simpa using h (mem_univ i)
  obtain ⟨i0, hi0, hmax⟩ := exists_max_image _ (fun i : Fin N => i.val) hne
  refine ⟨i0, (mem_filter.mp hi0).2, fun i hi => ?_⟩
  by_contra hne'
  have := hmax i (mem_filter.mpr ⟨mem_univ _, hne'⟩)
  rw [Fin.lt_def] at hi
  omega

/-- A nonzero vector has a first nonzero coordinate. -/
theorem exists_first_nonzero {b : ℕ} (τ : Fin b → F) (hτ : τ ≠ 0) :
    ∃ k0 : Fin b, τ k0 ≠ 0 ∧ ∀ k, k < k0 → τ k = 0 := by
  classical
  have hne : (univ.filter fun k : Fin b => τ k ≠ 0).Nonempty := by
    by_contra h
    apply hτ
    ext k
    rw [Finset.not_nonempty_iff_eq_empty, Finset.filter_eq_empty_iff] at h
    simpa using h (mem_univ k)
  obtain ⟨k0, hk0, hmin⟩ := exists_min_image _ (fun k : Fin b => k.val) hne
  refine ⟨k0, (mem_filter.mp hk0).2, fun k hk => ?_⟩
  by_contra hne'
  have := hmin k (mem_filter.mpr ⟨mem_univ _, hne'⟩)
  rw [Fin.lt_def] at hk
  omega

/-- The triangular step: restricted to the window starting at the last nonzero
coordinate `i0` of `a`, the map `T_a` is injective. -/
theorem rowMap_comp_window_injective {N b : ℕ} (a : Fin N → F) (i0 : Fin N)
    (h0 : a i0 ≠ 0) (hi : ∀ i, i0 < i → a i = 0) :
    Function.Injective ((rowMap N b a).comp (windowMap N b i0.val)) := by
  rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
  intro τ hτ
  by_contra hne
  obtain ⟨k0, hk0, hmin⟩ := exists_first_nonzero τ hne
  have hb : k0.val < b := k0.isLt
  obtain ⟨j, hj⟩ : ∃ j : Fin b, j.val = b - 1 - k0.val := ⟨⟨b - 1 - k0.val, by omega⟩, rfl⟩
  have hcoord := congrFun hτ j
  simp only [LinearMap.comp_apply, windowMap_apply, rowMap_apply, Pi.zero_apply,
    Matrix.vecMul, dotProduct] at hcoord
  rw [Finset.sum_eq_single i0] at hcoord
  · apply h0
    have hentry : toeplitz N b (window N b i0.val τ) i0 j = τ k0 := by
      rw [toeplitz_apply, window_apply_of_mem]
      · congr 1
        ext
        dsimp only
        omega
      · dsimp only
        omega
    rw [hentry] at hcoord
    exact (mul_eq_zero.mp hcoord).resolve_right hk0
  · intro i _ hne_i
    rcases lt_or_gt_of_ne hne_i with hlt | hgt
    · -- `i < i0`: the window entry is either outside the window or a coordinate
      -- of `τ` before `k0`, hence zero.
      rw [toeplitz_apply]
      by_cases hmem : i0.val ≤ i.val + (b - 1 - j.val) ∧ i.val + (b - 1 - j.val) < i0.val + b
      · rw [window_apply_of_mem _ _ _ hmem]
        have hk : (⟨i.val + (b - 1 - j.val) - i0.val, by omega⟩ : Fin b) < k0 := by
          rw [Fin.lt_def]
          rw [Fin.lt_def] at hlt
          dsimp only
          omega
        rw [hmin _ hk, mul_zero]
      · rw [window_apply_of_not_mem _ _ _ hmem, mul_zero]
    · rw [hi i hgt, zero_mul]
  · intro h
    exact absurd (mem_univ i0) h

/-- **Theorem A** (surjectivity): for every nonzero row `a`, `σ ↦ a Y(σ)` is onto `F^b`. -/
theorem rowMap_surjective {N b : ℕ} (a : Fin N → F) (ha : a ≠ 0) :
    Function.Surjective (rowMap N b a) := by
  obtain ⟨i0, h0, hi⟩ := exists_last_nonzero a ha
  have hinj := rowMap_comp_window_injective (b := b) a i0 h0 hi
  have hsurj := LinearMap.injective_iff_surjective.mp hinj
  intro t
  obtain ⟨τ, hτ⟩ := hsurj t
  exact ⟨windowMap N b i0.val τ, hτ⟩

theorem exists_seed_of_ne_zero {N b : ℕ} (a : Fin N → F) (ha : a ≠ 0) (t : Fin b → F) :
    ∃ σ : Fin (N + b - 1) → F, a ᵥ* toeplitz N b σ = t :=
  rowMap_surjective a ha t

/-! ## Rank characterization -/

/-- `rank M < g` for a `g × b` matrix iff some nonzero row combination vanishes. -/
theorem rank_lt_iff_exists_vecMul_eq_zero {g b : ℕ} (M : Matrix (Fin g) (Fin b) F) :
    M.rank < g ↔ ∃ l : Fin g → F, l ≠ 0 ∧ l ᵥ* M = 0 := by
  rw [← Matrix.rank_transpose, Matrix.rank]
  have hrn := LinearMap.finrank_range_add_finrank_ker (Mᵀ.mulVecLin)
  rw [Module.finrank_fin_fun] at hrn
  constructor
  · intro h
    have hker : LinearMap.ker Mᵀ.mulVecLin ≠ ⊥ := by
      intro hbot
      rw [hbot, finrank_bot] at hrn
      omega
    obtain ⟨l, hl, hl0⟩ := (Submodule.ne_bot_iff _).mp hker
    refine ⟨l, hl0, ?_⟩
    rw [LinearMap.mem_ker, Matrix.mulVecLin_apply, Matrix.mulVec_transpose] at hl
    exact hl
  · rintro ⟨l, hl0, hl⟩
    have hker : LinearMap.ker Mᵀ.mulVecLin ≠ ⊥ := by
      rw [Submodule.ne_bot_iff]
      refine ⟨l, ?_, hl0⟩
      rw [LinearMap.mem_ker, Matrix.mulVecLin_apply, Matrix.mulVec_transpose]
      exact hl
    have : Module.finrank F (LinearMap.ker Mᵀ.mulVecLin) ≠ 0 :=
      fun h => hker (Submodule.finrank_eq_zero.mp h)
    omega

/-- Full row rank of a `g × N` matrix is equivalent to injectivity of `l ↦ l ψ`. -/
theorem rank_eq_iff_vecMul_injective {g N : ℕ} (ψ : Matrix (Fin g) (Fin N) F) :
    ψ.rank = g ↔ ∀ l : Fin g → F, l ᵥ* ψ = 0 → l = 0 := by
  have hle : ψ.rank ≤ g := by
    have := Matrix.rank_le_card_height ψ
    simpa using this
  constructor
  · intro h l hl
    by_contra hne
    have : ψ.rank < g := (rank_lt_iff_exists_vecMul_eq_zero ψ).mpr ⟨l, hne, hl⟩
    omega
  · intro h
    by_contra hne
    have hlt : ψ.rank < g := lt_of_le_of_ne hle hne
    obtain ⟨l, hl0, hl⟩ := (rank_lt_iff_exists_vecMul_eq_zero ψ).mp hlt
    exact hl0 (h l hl)

/-- Surjectivity of `ψ : F^N → F^g` (as `v ↦ ψ v`) is equivalent to the row-injectivity
hypothesis used below. -/
theorem mulVec_surjective_iff_vecMul_injective {g N : ℕ} (ψ : Matrix (Fin g) (Fin N) F) :
    Function.Surjective ψ.mulVec ↔ ∀ l : Fin g → F, l ᵥ* ψ = 0 → l = 0 := by
  rw [← rank_eq_iff_vecMul_injective, Matrix.rank]
  show Function.Surjective ψ.mulVecLin ↔ _
  constructor
  · intro h
    have : LinearMap.range ψ.mulVecLin = ⊤ := by
      rw [LinearMap.range_eq_top]
      exact h
    rw [this, finrank_top, Module.finrank_fin_fun]
  · intro h
    have : LinearMap.range ψ.mulVecLin = ⊤ := by
      apply Submodule.eq_top_of_finrank_eq
      rw [h, Module.finrank_fin_fun]
    rw [← LinearMap.range_eq_top]
    exact this

/-! ## Theorem A: exact fiber counts and uniformity -/

section Counting

variable [Fintype F] [DecidableEq F]

/-- All fibers of a surjective linear map between finite modules have the same
cardinality, namely `|V| / |W|`. -/
theorem card_fiber_mul_card_of_surjective {V W : Type*} [AddCommGroup V] [AddCommGroup W]
    [Module F V] [Module F W] [Fintype V] [Fintype W] [DecidableEq W]
    (f : V →ₗ[F] W) (hf : Function.Surjective f) (t : W) :
    Fintype.card {v // f v = t} * Fintype.card W = Fintype.card V := by
  have key : ∀ t : W, Fintype.card {v // f v = t} = Fintype.card {v // f v = 0} := by
    intro t
    obtain ⟨v0, hv0⟩ := hf t
    refine Fintype.card_congr
      ⟨fun v => ⟨v.1 - v0, by simp [map_sub, v.2, hv0]⟩,
       fun v => ⟨v.1 + v0, by simp [map_add, v.2, hv0]⟩, ?_, ?_⟩
    · intro v; ext; simp
    · intro v; ext; simp
  have hsum := Fintype.card_congr (Equiv.sigmaFiberEquiv f)
  rw [Fintype.card_sigma] at hsum
  simp only [key] at hsum
  rw [Finset.sum_const, Finset.card_univ, smul_eq_mul] at hsum
  rw [key t, mul_comm]
  exact hsum

theorem card_seed (N b : ℕ) :
    Fintype.card (Fin (N + b - 1) → F) = Fintype.card F ^ (N + b - 1) := by
  simp

/-- **Theorem A** (fiber count): every value `t ∈ F^b` has exactly `q^(N-1)` seeds
with `a Y(σ) = t`. -/
theorem card_fiber_rowMap {N b : ℕ} (a : Fin N → F) (ha : a ≠ 0) (t : Fin b → F) :
    Fintype.card {σ : Fin (N + b - 1) → F // a ᵥ* toeplitz N b σ = t} =
      Fintype.card F ^ (N - 1) := by
  have hN : 1 ≤ N := by
    rcases Nat.eq_zero_or_pos N with h | h
    · subst h; exact absurd (funext fun i => i.elim0) ha
    · exact h
  have h := card_fiber_mul_card_of_surjective (rowMap N b a) (rowMap_surjective a ha) t
  simp only [rowMap_apply, Fintype.card_fun, Fintype.card_fin] at h
  have hq : 0 < Fintype.card F := Fintype.card_pos
  have hsplit : Fintype.card F ^ (N + b - 1) = Fintype.card F ^ (N - 1) * Fintype.card F ^ b := by
    rw [← pow_add]; congr 1; omega
  rw [hsplit] at h
  exact Nat.eq_of_mul_eq_mul_right (pow_pos hq b) h

/-- **Theorem A** (equation (3)): `Pr[a Y = 0] = q^{-b}`, as an exact rational ratio. -/
theorem prob_row_eq_zero {N b : ℕ} (a : Fin N → F) (ha : a ≠ 0) :
    (Fintype.card {σ : Fin (N + b - 1) → F // a ᵥ* toeplitz N b σ = 0} : ℚ) /
        Fintype.card (Fin (N + b - 1) → F) = 1 / (Fintype.card F : ℚ) ^ b := by
  have hN : 1 ≤ N := by
    rcases Nat.eq_zero_or_pos N with h | h
    · subst h; exact absurd (funext fun i => i.elim0) ha
    · exact h
  rw [card_fiber_rowMap a ha 0, card_seed]
  have hq : (0 : ℚ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  have hsplit : (Fintype.card F : ℚ) ^ (N + b - 1) =
      (Fintype.card F : ℚ) ^ (N - 1) * (Fintype.card F : ℚ) ^ b := by
    rw [← pow_add]; congr 1; omega
  push_cast
  rw [hsplit]
  field_simp

/-- Uniformity: every value of `F^b` is hit by the same number of seeds. -/
theorem card_fiber_rowMap_eq {N b : ℕ} (a : Fin N → F) (ha : a ≠ 0) (t t' : Fin b → F) :
    Fintype.card {σ : Fin (N + b - 1) → F // a ᵥ* toeplitz N b σ = t} =
      Fintype.card {σ : Fin (N + b - 1) → F // a ᵥ* toeplitz N b σ = t'} := by
  rw [card_fiber_rowMap a ha t, card_fiber_rowMap a ha t']

/-! ## Theorem B: quotient visibility (union bound) -/

/-- The finite counting statement behind Theorem B: the number of seeds for which
some nonzero functional `l` has `(lψ) Y(σ) = 0` is at most `(q^g - 1) q^(N-1)`.
Only a union bound is used; no independence between different `l` is assumed. -/
theorem card_vecMul_failure_le {N b g : ℕ} (ψ : Matrix (Fin g) (Fin N) F)
    (hψ : ∀ l : Fin g → F, l ᵥ* ψ = 0 → l = 0) :
    (univ.filter fun σ : Fin (N + b - 1) → F =>
        ∃ l : Fin g → F, l ≠ 0 ∧ l ᵥ* (ψ * toeplitz N b σ) = 0).card
      ≤ (Fintype.card F ^ g - 1) * Fintype.card F ^ (N - 1) := by
  classical
  calc (univ.filter fun σ : Fin (N + b - 1) → F =>
          ∃ l : Fin g → F, l ≠ 0 ∧ l ᵥ* (ψ * toeplitz N b σ) = 0).card
      ≤ ((univ.erase (0 : Fin g → F)).biUnion fun l =>
          univ.filter fun σ : Fin (N + b - 1) → F => (l ᵥ* ψ) ᵥ* toeplitz N b σ = 0).card := by
        apply card_le_card
        intro σ hσ
        simp only [mem_filter, mem_univ, true_and] at hσ
        obtain ⟨l, hl, hlσ⟩ := hσ
        simp only [mem_biUnion, mem_erase, mem_univ, and_true, mem_filter, true_and]
        exact ⟨l, hl, by rwa [Matrix.vecMul_vecMul]⟩
    _ ≤ ∑ l ∈ univ.erase (0 : Fin g → F),
          (univ.filter fun σ : Fin (N + b - 1) → F => (l ᵥ* ψ) ᵥ* toeplitz N b σ = 0).card :=
        card_biUnion_le
    _ = ∑ _l ∈ univ.erase (0 : Fin g → F), Fintype.card F ^ (N - 1) := by
        apply sum_congr rfl
        intro l hl
        rw [mem_erase] at hl
        rw [← Fintype.card_subtype]
        exact card_fiber_rowMap (l ᵥ* ψ) (fun h => hl.1 (hψ l h)) 0
    _ = (Fintype.card F ^ g - 1) * Fintype.card F ^ (N - 1) := by
        rw [sum_const, smul_eq_mul, card_erase_of_mem (mem_univ _), card_univ,
          Fintype.card_fun, Fintype.card_fin]

/-- **Theorem B** (equation (1), counting form): if `ψ` has full row rank `g`, then
`rank (ψ Y(σ)) < g` for at most `(q^g - 1) q^(N-1)` of the `q^(N+b-1)` seeds. -/
theorem card_rank_failure_le {N b g : ℕ} (ψ : Matrix (Fin g) (Fin N) F)
    (hψ : ∀ l : Fin g → F, l ᵥ* ψ = 0 → l = 0) :
    (univ.filter fun σ : Fin (N + b - 1) → F => (ψ * toeplitz N b σ).rank < g).card
      ≤ (Fintype.card F ^ g - 1) * Fintype.card F ^ (N - 1) := by
  classical
  have h := card_vecMul_failure_le (b := b) ψ hψ
  convert h using 2
  ext σ
  simp only [mem_filter, mem_univ, true_and]
  exact rank_lt_iff_exists_vecMul_eq_zero _

/-- **Theorem B** (equation (1), ratio form): `Pr[rank(ψ Y) < g] ≤ (q^g - 1) q^{-b}`. -/
theorem prob_rank_failure_le {N b g : ℕ} (ψ : Matrix (Fin g) (Fin N) F)
    (hψ : ∀ l : Fin g → F, l ᵥ* ψ = 0 → l = 0) (hN : 1 ≤ N) :
    ((univ.filter fun σ : Fin (N + b - 1) → F => (ψ * toeplitz N b σ).rank < g).card : ℚ) /
        Fintype.card (Fin (N + b - 1) → F)
      ≤ ((Fintype.card F : ℚ) ^ g - 1) / (Fintype.card F : ℚ) ^ b := by
  have h := card_rank_failure_le (b := b) ψ hψ
  have hq : (0 : ℚ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  have hq1 : 1 ≤ Fintype.card F ^ g := Nat.one_le_pow _ _ Fintype.card_pos
  rw [card_seed]
  have hsplit : (Fintype.card F : ℚ) ^ (N + b - 1) =
      (Fintype.card F : ℚ) ^ (N - 1) * (Fintype.card F : ℚ) ^ b := by
    rw [← pow_add]; congr 1; omega
  push_cast
  rw [hsplit, div_le_div_iff₀ (by positivity) (by positivity)]
  have h' : ((univ.filter fun σ : Fin (N + b - 1) → F =>
      (ψ * toeplitz N b σ).rank < g).card : ℚ) ≤
      ((Fintype.card F : ℚ) ^ g - 1) * (Fintype.card F : ℚ) ^ (N - 1) := by
    have := (Nat.cast_le (α := ℚ)).mpr h
    push_cast [Nat.cast_sub hq1] at this
    exact this
  calc ((univ.filter fun σ : Fin (N + b - 1) → F =>
          (ψ * toeplitz N b σ).rank < g).card : ℚ) * (Fintype.card F : ℚ) ^ b
      ≤ ((Fintype.card F : ℚ) ^ g - 1) * (Fintype.card F : ℚ) ^ (N - 1) *
          (Fintype.card F : ℚ) ^ b := by
        apply mul_le_mul_of_nonneg_right h' (by positivity)
    _ = ((Fintype.card F : ℚ) ^ g - 1) *
          ((Fintype.card F : ℚ) ^ (N - 1) * (Fintype.card F : ℚ) ^ b) := by ring

/-- The strict form `(q^g - 1) q^{-b} < q^{g-b}` of equation (1). -/
theorem prob_rank_failure_lt {N b g : ℕ} (ψ : Matrix (Fin g) (Fin N) F)
    (hψ : ∀ l : Fin g → F, l ᵥ* ψ = 0 → l = 0) (hN : 1 ≤ N) :
    ((univ.filter fun σ : Fin (N + b - 1) → F => (ψ * toeplitz N b σ).rank < g).card : ℚ) /
        Fintype.card (Fin (N + b - 1) → F)
      < (Fintype.card F : ℚ) ^ g / (Fintype.card F : ℚ) ^ b := by
  refine lt_of_le_of_lt (prob_rank_failure_le ψ hψ hN) ?_
  have hq : (0 : ℚ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  apply div_lt_div_of_pos_right _ (by positivity)
  linarith

/-! ## Rank-`r` variant (equation (4)) -/

/-- **Theorem B, rank-`r` form**: if a fixed quotient `π` makes `π ψ` of full row rank
`r`, then `rank (ψ Y(σ)) < r` for at most `(q^r - 1) q^(N-1)` seeds.  The event is
stated for `ψ Y` itself; the quotient `π` is only used in the proof. -/
theorem card_rank_lt_r_le {N b g' r : ℕ} (ψ : Matrix (Fin g') (Fin N) F)
    (π : Matrix (Fin r) (Fin g') F)
    (hπψ : ∀ l : Fin r → F, l ᵥ* (π * ψ) = 0 → l = 0) :
    (univ.filter fun σ : Fin (N + b - 1) → F => (ψ * toeplitz N b σ).rank < r).card
      ≤ (Fintype.card F ^ r - 1) * Fintype.card F ^ (N - 1) := by
  classical
  refine le_trans (card_le_card ?_) (card_rank_failure_le (b := b) (π * ψ) hπψ)
  intro σ hσ
  simp only [mem_filter, mem_univ, true_and] at hσ ⊢
  calc (π * ψ * toeplitz N b σ).rank ≤ (ψ * toeplitz N b σ).rank := by
        rw [Matrix.mul_assoc]; exact Matrix.rank_mul_le_right _ _
    _ < r := hσ

/-- A matrix of rank at least `r` has a fixed rank-`r` quotient `π` with `π ψ` of
full row rank. -/
theorem exists_quotient_of_le_rank {g' N : ℕ} (ψ : Matrix (Fin g') (Fin N) F) (r : ℕ)
    (hr : r ≤ ψ.rank) :
    ∃ π : Matrix (Fin r) (Fin g') F, ∀ l : Fin r → F, l ᵥ* (π * ψ) = 0 → l = 0 := by
  classical
  rw [Matrix.rank_eq_finrank_span_row] at hr
  obtain ⟨v, hv⟩ := exists_linearIndependent_of_le_finrank hr
  choose c hc using fun k => (Submodule.mem_span_range_iff_exists_fun (R := F)).mp (v k).2
  refine ⟨Matrix.of c, fun l hl => ?_⟩
  have hrow : ∀ k, (Matrix.of c * ψ) k = (v k : Fin N → F) := by
    intro k
    rw [← hc k]
    ext j
    simp [Matrix.mul_apply, Finset.sum_apply, Matrix.row]
  have hsum : ∑ k, l k • v k = 0 := by
    apply Subtype.ext
    simp only [Submodule.coe_sum, Submodule.coe_smul, ZeroMemClass.coe_zero]
    rw [Matrix.vecMul_eq_sum] at hl
    simpa [hrow] using hl
  ext k
  exact Fintype.linearIndependent_iff.mp hv l hsum k

end Counting

/-! ## Theorem C: full column rank of the start (via transposition) -/

/-- Seed reversal `σ'_m = σ_{N+b-2-m}`, mapping seeds of the `N × b` panel to seeds of
the `b × N` panel. -/
def rev (N b : ℕ) (σ : Fin (N + b - 1) → F) : Fin (b + N - 1) → F :=
  fun m => σ ⟨N + b - 2 - m.val, by have := m.isLt; omega⟩

theorem rev_rev {N b : ℕ} (σ : Fin (N + b - 1) → F) : rev b N (rev N b σ) = σ := by
  ext m
  simp only [rev]
  congr 1
  ext
  simp only
  have := m.isLt
  omega

theorem rev_bijective (N b : ℕ) : Function.Bijective (rev (F := F) N b) :=
  Function.bijective_iff_has_inverse.mpr ⟨rev b N, rev_rev, rev_rev⟩

/-- The transpose of the `N × b` Toeplitz panel is the `b × N` Toeplitz panel of the
reversed seed, in the same convention. -/
theorem transpose_toeplitz {N b : ℕ} (σ : Fin (N + b - 1) → F) :
    (toeplitz N b σ)ᵀ = toeplitz b N (rev N b σ) := by
  ext j i
  simp only [Matrix.transpose_apply, toeplitz_apply, rev]
  congr 1
  ext
  simp only
  have := i.isLt
  have := j.isLt
  omega

/-- **Theorem C** (surjectivity): for every nonzero column `c`, `σ ↦ Y(σ) c` is onto `F^N`. -/
theorem colMap_surjective {N b : ℕ} (c : Fin b → F) (hc : c ≠ 0) :
    Function.Surjective (fun σ : Fin (N + b - 1) → F => toeplitz N b σ *ᵥ c) := by
  intro t
  obtain ⟨σ', hσ'⟩ := rowMap_surjective (N := b) (b := N) c hc t
  refine ⟨rev b N σ', ?_⟩
  show toeplitz N b (rev b N σ') *ᵥ c = t
  rw [← Matrix.vecMul_transpose, transpose_toeplitz, rev_rev]
  exact hσ'

/-- `rank M < b` for an `N × b` matrix iff some nonzero column combination vanishes. -/
theorem rank_lt_iff_exists_mulVec_eq_zero {N b : ℕ} (M : Matrix (Fin N) (Fin b) F) :
    M.rank < b ↔ ∃ c : Fin b → F, c ≠ 0 ∧ M *ᵥ c = 0 := by
  rw [← Matrix.rank_transpose, rank_lt_iff_exists_vecMul_eq_zero]
  simp only [Matrix.vecMul_transpose]

section ColumnCounting

variable [Fintype F] [DecidableEq F]

/-- **Theorem C** (fiber count): every `t ∈ F^N` has exactly `q^(b-1)` seeds with
`Y(σ) c = t`. -/
theorem card_fiber_colMap {N b : ℕ} (c : Fin b → F) (hc : c ≠ 0) (t : Fin N → F) :
    Fintype.card {σ : Fin (N + b - 1) → F // toeplitz N b σ *ᵥ c = t} =
      Fintype.card F ^ (b - 1) := by
  rw [← card_fiber_rowMap (N := b) (b := N) c hc t]
  apply Fintype.card_congr
  refine ⟨fun σ => ⟨rev N b σ.1, ?_⟩, fun σ' => ⟨rev b N σ'.1, ?_⟩, ?_, ?_⟩
  · rw [← transpose_toeplitz, Matrix.vecMul_transpose]; exact σ.2
  · rw [← Matrix.vecMul_transpose, transpose_toeplitz, rev_rev]; exact σ'.2
  · intro σ; ext; simp [rev_rev]
  · intro σ'; ext; simp [rev_rev]

/-- **Theorem C** (equation (5), counting form): `rank Y(σ) < b` for at most
`(q^b - 1) q^(b-1)` of the `q^(N+b-1)` seeds. -/
theorem card_column_rank_failure_le (N b : ℕ) :
    (univ.filter fun σ : Fin (N + b - 1) → F => (toeplitz N b σ).rank < b).card
      ≤ (Fintype.card F ^ b - 1) * Fintype.card F ^ (b - 1) := by
  classical
  have h := card_rank_failure_le (N := b) (b := N) (g := b) (1 : Matrix (Fin b) (Fin b) F)
    (fun l hl => by simpa using hl)
  refine le_trans (le_of_eq ?_) h
  apply Finset.card_bij (fun σ _ => rev N b σ)
  · intro σ hσ
    simp only [mem_filter, mem_univ, true_and, Matrix.one_mul] at hσ ⊢
    rw [← transpose_toeplitz, Matrix.rank_transpose]
    exact hσ
  · intro σ₁ _ σ₂ _ h
    exact (rev_bijective N b).1 h
  · intro σ' hσ'
    refine ⟨rev b N σ', ?_, rev_rev σ'⟩
    simp only [mem_filter, mem_univ, true_and, Matrix.one_mul] at hσ' ⊢
    rw [← Matrix.rank_transpose, transpose_toeplitz, rev_rev]
    exact hσ'

/-- **Theorem C** (equation (5), ratio form): `Pr[rank Y < b] ≤ (q^b - 1) q^{-N}`. -/
theorem prob_column_rank_failure_le (N b : ℕ) (hb : 1 ≤ b) :
    ((univ.filter fun σ : Fin (N + b - 1) → F => (toeplitz N b σ).rank < b).card : ℚ) /
        Fintype.card (Fin (N + b - 1) → F)
      ≤ ((Fintype.card F : ℚ) ^ b - 1) / (Fintype.card F : ℚ) ^ N := by
  have h := card_column_rank_failure_le (F := F) N b
  have hq : (0 : ℚ) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  have hq1 : 1 ≤ Fintype.card F ^ b := Nat.one_le_pow _ _ Fintype.card_pos
  rw [card_seed]
  have hsplit : (Fintype.card F : ℚ) ^ (N + b - 1) =
      (Fintype.card F : ℚ) ^ (b - 1) * (Fintype.card F : ℚ) ^ N := by
    rw [← pow_add]; congr 1; omega
  push_cast
  rw [hsplit, div_le_div_iff₀ (by positivity) (by positivity)]
  have h' : ((univ.filter fun σ : Fin (N + b - 1) → F =>
      (toeplitz N b σ).rank < b).card : ℚ) ≤
      ((Fintype.card F : ℚ) ^ b - 1) * (Fintype.card F : ℚ) ^ (b - 1) := by
    have := (Nat.cast_le (α := ℚ)).mpr h
    push_cast [Nat.cast_sub hq1] at this
    exact this
  calc ((univ.filter fun σ : Fin (N + b - 1) → F =>
          (toeplitz N b σ).rank < b).card : ℚ) * (Fintype.card F : ℚ) ^ N
      ≤ ((Fintype.card F : ℚ) ^ b - 1) * (Fintype.card F : ℚ) ^ (b - 1) *
          (Fintype.card F : ℚ) ^ N := by
        apply mul_le_mul_of_nonneg_right h' (by positivity)
    _ = ((Fintype.card F : ℚ) ^ b - 1) *
          ((Fintype.card F : ℚ) ^ (b - 1) * (Fintype.card F : ℚ) ^ N) := by ring

end ColumnCounting

/-! ## Theorem D: zero-primary transfer -/

/-- **Theorem D** (equation (6)): if `ψ A = 0`, then for `Q = ∑_{i ≤ d} A^i Y F_i`
one has `ψ Q = (ψ Y) F_0`.  This is a deterministic identity with no probabilistic
content and no claim about `F_0`. -/
theorem zero_primary_transfer {N b g k : ℕ} (A : Matrix (Fin N) (Fin N) F)
    (ψ : Matrix (Fin g) (Fin N) F) (Y : Matrix (Fin N) (Fin b) F)
    (Fc : ℕ → Matrix (Fin b) (Fin k) F) (d : ℕ) (hψA : ψ * A = 0) :
    ψ * (∑ i ∈ range (d + 1), A ^ i * Y * Fc i) = ψ * Y * Fc 0 := by
  rw [Matrix.mul_sum, Finset.sum_range_succ']
  have hzero : ∀ i ∈ range d, ψ * (A ^ (i + 1) * Y * Fc (i + 1)) = 0 := by
    intro i _
    rw [pow_succ', ← Matrix.mul_assoc, ← Matrix.mul_assoc, ← Matrix.mul_assoc, hψA,
      Matrix.zero_mul, Matrix.zero_mul, Matrix.zero_mul]
  rw [Finset.sum_eq_zero hzero, zero_add, pow_zero, Matrix.one_mul, Matrix.mul_assoc]

/-- The same identity with the Toeplitz panel substituted. -/
theorem zero_primary_transfer_toeplitz {N b g k : ℕ} (A : Matrix (Fin N) (Fin N) F)
    (ψ : Matrix (Fin g) (Fin N) F) (σ : Fin (N + b - 1) → F)
    (Fc : ℕ → Matrix (Fin b) (Fin k) F) (d : ℕ) (hψA : ψ * A = 0) :
    ψ * (∑ i ∈ range (d + 1), A ^ i * toeplitz N b σ * Fc i) = ψ * toeplitz N b σ * Fc 0 :=
  zero_primary_transfer A ψ _ Fc d hψA

/-! ## Theorem E: conditional retries

The Toeplitz bound enters only through the hypothesis `μ Vᶜ ≤ ε` on a visibility event
`V`; the conditional success bound `ρ` is an explicit hypothesis that nothing in this
file produces. -/

section Retry

open MeasureTheory ProbabilityTheory ENNReal

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **Theorem E, one attempt (equation (7))**: if visibility fails with probability at most
`ε` and the later stage succeeds, conditional on visibility, with probability at least `ρ`,
then the attempt succeeds with probability at least `(1 - ε) ρ`. -/
theorem one_attempt_success_ge (μ : Measure Ω) [IsProbabilityMeasure μ] {V S : Set Ω}
    (hV : MeasurableSet V) {ε ρ : ℝ≥0∞} (hε : μ Vᶜ ≤ ε) (hρ : ρ ≤ μ[S | V]) :
    (1 - ε) * ρ ≤ μ S := by
  have hVge : 1 - ε ≤ μ V := by
    have h1 : μ V = 1 - μ Vᶜ := by
      rw [prob_compl_eq_one_sub hV, ENNReal.sub_sub_cancel ENNReal.one_ne_top prob_le_one]
    rw [h1]
    exact tsub_le_tsub_left hε 1
  calc (1 - ε) * ρ ≤ μ V * μ[S | V] := mul_le_mul' hVge hρ
    _ = μ (V ∩ S) := by rw [mul_comm]; exact cond_mul_eq_inter hV S μ
    _ ≤ μ S := measure_mono Set.inter_subset_right

/-- The event that the first `k` attempts all failed, for success events `S 0, S 1, …`. -/
def failures (S : ℕ → Set Ω) : ℕ → Set Ω
  | 0 => Set.univ
  | k + 1 => failures S k ∩ (S k)ᶜ

theorem measurableSet_failures {S : ℕ → Set Ω} (hS : ∀ k, MeasurableSet (S k)) (k : ℕ) :
    MeasurableSet (failures S k) := by
  induction k with
  | zero => exact MeasurableSet.univ
  | succ k ih => exact ih.inter (hS k).compl

theorem failures_succ_subset (S : ℕ → Set Ω) (k : ℕ) : failures S (k + 1) ⊆ failures S k :=
  Set.inter_subset_left

/-- One retry step: if attempt `k` succeeds with conditional probability at least `p`
given that every earlier attempt failed, the failure probability shrinks by `(1 - p)`. -/
theorem measure_failures_succ_le (μ : Measure Ω) [IsFiniteMeasure μ] {S : ℕ → Set Ω}
    (hS : ∀ k, MeasurableSet (S k)) {p : ℝ≥0∞} (k : ℕ) (hp : p ≤ μ[S k | failures S k]) :
    μ (failures S (k + 1)) ≤ (1 - p) * μ (failures S k) := by
  have hF := measurableSet_failures hS k
  have hsplit : μ (failures S k ∩ S k) + μ (failures S k \ S k) = μ (failures S k) :=
    measure_inter_add_sdiff _ (hS k)
  have hinter : p * μ (failures S k) ≤ μ (failures S k ∩ S k) := by
    rw [← cond_mul_eq_inter hF (S k) μ]
    exact mul_le_mul' hp le_rfl
  have hfin : μ (failures S k) ≠ ∞ := measure_ne_top μ _
  have hdiff : failures S (k + 1) = failures S k \ S k := Set.ext fun _ => Iff.rfl
  have hsub : μ (failures S k \ S k) = μ (failures S k) - μ (failures S k ∩ S k) := by
    conv_rhs => rw [← hsplit]
    rw [ENNReal.add_sub_cancel_left (measure_ne_top μ _)]
  rw [hdiff, ENNReal.sub_mul (fun _ _ => hfin), one_mul, hsub]
  exact tsub_le_tsub_left hinter _

/-- **Theorem E, tail bound (equation (8))**: if every attempt succeeds with conditional
probability at least `p` given that all previous attempts failed, then the first `k`
attempts all fail with probability at most `(1 - p)^k`. -/
theorem measure_failures_le (μ : Measure Ω) [IsProbabilityMeasure μ] {S : ℕ → Set Ω}
    (hS : ∀ k, MeasurableSet (S k)) {p : ℝ≥0∞} (hp : ∀ k, p ≤ μ[S k | failures S k])
    (k : ℕ) : μ (failures S k) ≤ (1 - p) ^ k := by
  induction k with
  | zero => simp [failures]
  | succ k ih =>
    calc μ (failures S (k + 1)) ≤ (1 - p) * μ (failures S k) :=
          measure_failures_succ_le μ hS k (hp k)
      _ ≤ (1 - p) * (1 - p) ^ k := mul_le_mul' le_rfl ih
      _ = (1 - p) ^ (k + 1) := (pow_succ' _ _).symm

/-- **Theorem E, expected attempts (equation (8))**: if `T` counts attempts and `T > k`
forces the first `k` attempts to fail, then `E[T] ≤ 1/p`. -/
theorem lintegral_attempts_le (μ : Measure Ω) [IsProbabilityMeasure μ] {S : ℕ → Set Ω}
    (hS : ∀ k, MeasurableSet (S k)) {p : ℝ≥0∞} (hp1 : p ≤ 1)
    (hp : ∀ k, p ≤ μ[S k | failures S k])
    (T : Ω → ℕ) (hT : Measurable T) (hTS : ∀ k, {ω | k < T ω} ⊆ failures S k) :
    ∫⁻ ω, (T ω : ℝ≥0∞) ∂μ ≤ p⁻¹ := by
  have hmeas : ∀ k : ℕ, MeasurableSet {ω | k < T ω} := fun k =>
    measurableSet_lt measurable_const hT
  have hrepr : ∀ ω, (T ω : ℝ≥0∞) = ∑' k : ℕ, {ω | k < T ω}.indicator (1 : Ω → ℝ≥0∞) ω := by
    intro ω
    have hsupp : ∀ k ∉ Finset.range (T ω),
        {ω | k < T ω}.indicator (1 : Ω → ℝ≥0∞) ω = 0 := by
      intro k hk
      simp only [Finset.mem_range, not_lt] at hk
      exact Set.indicator_of_notMem (s := {ω | k < T ω}) (a := ω) (not_lt.mpr hk) (1 : Ω → ℝ≥0∞)
    have hone : ∀ k ∈ Finset.range (T ω),
        {ω | k < T ω}.indicator (1 : Ω → ℝ≥0∞) ω = 1 := by
      intro k hk
      exact Set.indicator_of_mem (s := {ω | k < T ω}) (a := ω)
        (show ω ∈ {ω | k < T ω} from Finset.mem_range.mp hk) (1 : Ω → ℝ≥0∞)
    rw [tsum_eq_sum hsupp, Finset.sum_congr rfl hone, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul, mul_one]
  calc ∫⁻ ω, (T ω : ℝ≥0∞) ∂μ
      = ∫⁻ ω, ∑' k : ℕ, {ω | k < T ω}.indicator (1 : Ω → ℝ≥0∞) ω ∂μ := lintegral_congr hrepr
    _ = ∑' k : ℕ, ∫⁻ ω, {ω | k < T ω}.indicator (1 : Ω → ℝ≥0∞) ω ∂μ :=
        lintegral_tsum fun k => (measurable_one.indicator (hmeas k)).aemeasurable
    _ = ∑' k : ℕ, μ {ω | k < T ω} := tsum_congr fun k => lintegral_indicator_one (hmeas k)
    _ ≤ ∑' k : ℕ, (1 - p) ^ k := ENNReal.tsum_le_tsum fun k =>
        (measure_mono (hTS k)).trans (measure_failures_le μ hS hp k)
    _ = (1 - (1 - p))⁻¹ := ENNReal.tsum_geometric _
    _ = p⁻¹ := by rw [ENNReal.sub_sub_cancel ENNReal.one_ne_top hp1]

/-- **Theorem E, combined form**: if at every retry, conditional on all previous failures,
visibility fails with probability at most `ε` and success given visibility is at least `ρ`,
then the failure tail is bounded by `(1 - (1 - ε) ρ)^k`.  With `ε = (q^g - 1) q^{-b}` from
Theorem B this is equation (8) with `p = (1 - ε) ρ`. -/
theorem measure_failures_le_of_visibility (μ : Measure Ω) [IsProbabilityMeasure μ]
    {S V : ℕ → Set Ω} (hS : ∀ k, MeasurableSet (S k)) (hV : ∀ k, MeasurableSet (V k))
    {ε ρ : ℝ≥0∞}
    (hε : ∀ k, μ[(V k)ᶜ | failures S k] ≤ ε)
    (hρ : ∀ k, ρ ≤ μ[|failures S k][S k | V k]) (k : ℕ) :
    μ (failures S k) ≤ (1 - (1 - ε) * ρ) ^ k := by
  apply measure_failures_le μ hS
  intro k
  by_cases h0 : μ (failures S k) = 0
  · have hzero : μ[|failures S k] = 0 := cond_eq_zero_of_meas_eq_zero h0
    have hρ0 : ρ = 0 := by
      have hρ' := hρ k
      rw [hzero] at hρ'
      simpa [ProbabilityTheory.cond] using hρ'
    rw [hρ0, mul_zero]
    exact zero_le
  · have := cond_isProbabilityMeasure (μ := μ) h0
    exact one_attempt_success_ge μ[|failures S k] (hV k) (hε k) (hρ k)

end Retry

/-! ## Kernel-checked convention examples on supplied fixtures

These are decided by the kernel with `decide` (no compiled-code evaluation) and pin the index
convention against the fixture file `data/toeplitz_right_start/instances.json`. -/

section Examples

/-- The zero-primary fixture seed `[1,0,1,1,0,1]` with `N = 4`, `b = 3`. -/
example :
    toeplitz 4 3 (![1, 0, 1, 1, 0, 1] : Fin 6 → ZMod 2) =
      !![1, 0, 1; 1, 1, 0; 0, 1, 1; 1, 0, 1] := by
  decide

/-- The `N = 3`, `b = 2` seed `[1,0,1,1]`: row `i` is the reversed window `σ_{i+1}, σ_i`. -/
example :
    toeplitz 3 2 (![1, 0, 1, 1] : Fin 4 → ZMod 2) = !![0, 1; 1, 0; 1, 1] := by
  decide

/-- A full-rank panel that is invisible to the fixed quotient `ψ = [1, 0, 1]`
(seed `[0,1,0,1]`, `N = 3`, `b = 2`): full column rank does not imply visibility. -/
example :
    (!![1, 0, 1] : Matrix (Fin 1) (Fin 3) (ZMod 2)) *
        toeplitz 3 2 (![0, 1, 0, 1] : Fin 4 → ZMod 2) = 0 ∧
      toeplitz 3 2 (![0, 1, 0, 1] : Fin 4 → ZMod 2) *ᵥ ![1, 0] ≠ 0 ∧
      toeplitz 3 2 (![0, 1, 0, 1] : Fin 4 → ZMod 2) *ᵥ ![0, 1] ≠ 0 ∧
      toeplitz 3 2 (![0, 1, 0, 1] : Fin 4 → ZMod 2) *ᵥ ![1, 1] ≠ 0 := by
  decide

end Examples

end ToeplitzStart
