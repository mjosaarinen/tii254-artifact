import Mathlib.LinearAlgebra.Projection
import Mathlib.LinearAlgebra.Basis.VectorSpace

/-! Coordinate complements from a certified information set. An invertible
restriction is explicit input; dimensions alone are never used to infer it. -/

namespace ProjectedKernelRigidity

open LinearMap Submodule

variable {F V P Y : Type*} [Field F]
variable [AddCommGroup V] [Module F V]
variable [AddCommGroup P] [Module F P]
variable [AddCommGroup Y] [Module F Y]

structure CoordinateInformationSet (L : Submodule F V) (coordinates : V →ₗ[F] P) where
  information : L ≃ₗ[F] P
  agreement : information.toLinearMap = coordinates.comp L.subtype

namespace CoordinateInformationSet

variable {L : Submodule F V} {coordinates : V →ₗ[F] P}

def retract (data : CoordinateInformationSet L coordinates) : V →ₗ[F] L :=
  data.information.symm.toLinearMap.comp coordinates

theorem retract_subtype (data : CoordinateInformationSet L coordinates) (l : L) :
    data.retract l = l := by
  have h := LinearMap.congr_fun data.agreement l
  change data.information l = coordinates l at h
  change data.information.symm (coordinates l) = l
  rw [← h]
  exact data.information.symm_apply_apply l

theorem retract_ker (data : CoordinateInformationSet L coordinates) :
    data.retract.ker = coordinates.ker := by
  ext x
  change data.information.symm (coordinates x) = 0 ↔ coordinates x = 0
  constructor
  · intro h
    have hh := congrArg data.information h
    simpa using hh
  · intro h
    simp [h]

theorem isCompl (data : CoordinateInformationSet L coordinates) :
    IsCompl L coordinates.ker := by
  have h := LinearMap.isCompl_of_proj (data.retract_subtype)
  rw [data.retract_ker] at h
  exact h

/-- The quotient is identified with the coordinate complement, over any field. -/
noncomputable def quotientEquiv (data : CoordinateInformationSet L coordinates) :
    (V ⧸ L) ≃ₗ[F] coordinates.ker :=
  L.quotientEquivOfIsCompl coordinates.ker data.isCompl

def remove (data : CoordinateInformationSet L coordinates) : V →ₗ[F] V :=
  LinearMap.id - L.subtype.comp data.retract

theorem remove_coordinates (data : CoordinateInformationSet L coordinates) (x : V) :
    coordinates (data.remove x) = 0 := by
  have h := LinearMap.congr_fun data.agreement (data.retract x)
  change data.information (data.retract x) = coordinates (data.retract x) at h
  have hh : coordinates (data.retract x) = coordinates x := by
    rw [← h]
    exact data.information.apply_symm_apply (coordinates x)
  change coordinates (x - (data.retract x : V)) = 0
  rw [map_sub, hh, sub_self]

theorem remove_preserves_equation (data : CoordinateInformationSet L coordinates)
    (E : V →ₗ[F] Y) (hL : L ≤ E.ker) (x : V) :
    E (data.remove x) = E x := by
  have hz : E (data.retract x) = 0 := LinearMap.mem_ker.mp (hL (data.retract x).property)
  change E (x - (data.retract x : V)) = E x
  rw [map_sub, hz, sub_zero]

theorem remove_mem_kernel_iff (data : CoordinateInformationSet L coordinates)
    (E : V →ₗ[F] Y) (hL : L ≤ E.ker) (x : V) :
    data.remove x ∈ E.ker ↔ x ∈ E.ker := by
  simp only [LinearMap.mem_ker, data.remove_preserves_equation E hL]

/-- The quotient equation agrees with literal replay on the complement. -/
theorem quotient_equation_replay (data : CoordinateInformationSet L coordinates)
    (E : V →ₗ[F] Y) (hL : L ≤ E.ker) (x : coordinates.ker) :
    L.liftQ E hL (data.quotientEquiv.symm x) = E x := by
  change L.liftQ E hL (L.mkQ (x : V)) = E x
  exact Submodule.liftQ_apply _ _ _

end CoordinateInformationSet
end ProjectedKernelRigidity
