import ProjectedKernelRigidity.ObserverCompletenessFinite

/-! Literal verification on a full basis and the meaning of certificate
terminals. This formalizes identities, not a Python/Rust implementation. -/

namespace ProjectedKernelRigidity

open LinearMap Submodule Module

variable {F V Y Z ι : Type*} [Field F]
variable [AddCommGroup V] [Module F V]
variable [AddCommGroup Y] [Module F Y]
variable [AddCommGroup Z] [Module F Z]

def certificateOfBasis (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z)
    (basis : Basis ι F V) (H : Y →ₗ[F] Z ⧸ R)
    (checks : ∀ i, H (E (basis i)) = R.mkQ (O (basis i))) :
    ObserverCompletenessCertificate E O R where
  factor := H
  identity := basis.ext checks

def dualCertificateOfBasis (E : V →ₗ[F] Y) (O : V →ₗ[F] Z) (R : Submodule F Z)
    (basis : Basis ι F R.dualAnnihilator)
    (L : R.dualAnnihilator →ₗ[F] Module.Dual F Y)
    (checks : ∀ i, E.dualMap (L (basis i)) = O.dualMap (basis i)) :
    ObserverDualCertificate E O R where
  lift := L
  identity := basis.ext checks

theorem certificate_excludes_escape {E : V →ₗ[F] Y} {O : V →ₗ[F] Z}
    {R : Submodule F Z} (certificate : ObserverCompletenessCertificate E O R)
    (escape : ObserverEscapeWitness E O R) : False := by
  apply escape.escapes
  exact certificate.sound ⟨escape.source, LinearMap.mem_ker.mpr escape.equation, rfl⟩

theorem completeness_mono {E : V →ₗ[F] Y} {O : V →ₗ[F] Z}
    {R R' : Submodule F Z} (h : Submodule.map O E.ker ≤ R) (hRR : R ≤ R') :
    Submodule.map O E.ker ≤ R' := le_trans h hRR

/-- An escape is local to the subspace named in its certificate: it is
preserved under shrinking that subspace, not under arbitrary enlargement. -/
def escapeOfSmallerSubspace {E : V →ₗ[F] Y} {O : V →ₗ[F] Z}
    {R R' : Submodule F Z} (escape : ObserverEscapeWitness E O R) (hRR : R' ≤ R) :
    ObserverEscapeWitness E O R' where
  source := escape.source
  equation := escape.equation
  escapes := fun h => escape.escapes (hRR h)

end ProjectedKernelRigidity
