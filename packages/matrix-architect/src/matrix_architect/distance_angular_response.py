"""Scalar angular/SAM response in squared pair-distance descriptors.

The potential is U(t) + A(t).z + z.B.z/2.  Cartesian gradients and Hessians
include the complete descriptor chain rule, including residual force terms.
The angular section may be periodic in the geometry although U is represented
in its invariant distance descriptor. This is a local, finite-data response;
it makes no collision or dissociation-asymptote claim.
"""
from dataclasses import dataclass
import numpy as np


def squared_pair_distance_jet(coordinates, pairs):
    """Exact value, Cartesian first and second derivatives of squared distances."""
    x=np.asarray(coordinates,float)
    if x.ndim!=2 or x.shape[1]!=3 or not np.all(np.isfinite(x)):
        raise ValueError('finite N by 3 coordinates required')
    pairs=tuple(tuple(p) for p in pairs)
    if not pairs or any(len(p)!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<len(x) for i in p) or p[0]==p[1] for p in pairs):
        raise ValueError('distinct valid atom indices required for each pair')
    if len({tuple(sorted(p)) for p in pairs})!=len(pairs):
        raise ValueError('duplicate distance descriptor')
    values=np.empty(len(pairs));J=np.zeros((len(pairs),x.size));H=np.zeros((len(pairs),x.size,x.size))
    for k,(i,j) in enumerate(pairs):
        delta=x[i]-x[j];values[k]=delta@delta
        J[k,3*i:3*i+3]=2*delta;J[k,3*j:3*j+3]=-2*delta
        for a,b,sign in ((i,i,1),(j,j,1),(i,j,-1),(j,i,-1)):
            H[k,3*a:3*a+3,3*b:3*b+3]=2*sign*np.eye(3)
    return values,J,H


def fit_anchored_scalar_polynomial(reference_value, reference_first, observations, *, reference_second=None):
    """Minimal Hermite polynomial from explicitly supplied value/first rows.

    Observations are (coordinate, derivative order, value). No validation data
    or potential callback is accepted. A second reference derivative can be
    supplied to preserve the physical fourth angular jet when t=1-cos(phi).
    """
    anchors=[float(reference_value),float(reference_first)]
    if reference_second is not None:anchors.append(float(reference_second)/2)
    rows=tuple(observations);n=len(anchors)
    if not rows or not np.all(np.isfinite(anchors)):
        raise ValueError('finite anchors and nonempty observations required')
    matrix=[];target=[]
    for q,order,value in rows:
        if not isinstance(order,(int,np.integer)) or order not in (0,1) or isinstance(order,(bool,np.bool_)) or not np.all(np.isfinite([q,value])):
            raise ValueError('finite value/first derivative observation required')
        matrix.append([q**k if order==0 else k*q**(k-1) for k in range(n,n+len(rows))])
        base=np.polynomial.polynomial.polyval(q,np.polynomial.polynomial.polyder(anchors,order))
        target.append(value-base)
    matrix=np.asarray(matrix);scale=np.linalg.norm(matrix,axis=0)
    if np.any(scale<=0):raise ValueError('unobserved scalar coefficient')
    scaled=matrix/scale
    if np.linalg.matrix_rank(scaled,tol=1e-11)<len(rows):
        raise ValueError('unidentified scalar Hermite response')
    return np.r_[anchors,np.linalg.solve(scaled,target)/scale]


@dataclass(frozen=True)
class DistanceAngularResponse:
    pairs: tuple
    reference_squared_distances: np.ndarray
    active_scale: float
    angular_coefficients: np.ndarray
    force_coefficients: np.ndarray
    passive_quadratic: np.ndarray

    def __post_init__(self):
        pairs=tuple(tuple(p) for p in self.pairs);object.__setattr__(self,'pairs',pairs)
        if not np.isfinite(self.active_scale) or self.active_scale<=0:
            raise ValueError('positive active squared-distance scale required')
        for name in ('reference_squared_distances','angular_coefficients','force_coefficients','passive_quadratic'):
            v=np.array(getattr(self,name),float,copy=True)
            if not np.all(np.isfinite(v)):raise ValueError('finite response coefficients required')
            v.flags.writeable=False;object.__setattr__(self,name,v)
        m=len(pairs)-1
        if (m<1 or self.reference_squared_distances.shape!=(m+1,) or np.any(self.reference_squared_distances<=0)
                or self.angular_coefficients.ndim!=1 or not len(self.angular_coefficients)
                or self.force_coefficients.ndim!=2 or not len(self.force_coefficients)
                or self.force_coefficients.shape[1]!=m or self.passive_quadratic.shape!=(m,m)
                or not np.allclose(self.passive_quadratic,self.passive_quadratic.T,rtol=0,atol=1e-10)):
            raise ValueError('incompatible scalar distance-response dimensions')
        # Validate pair syntax without invoking a molecular potential.
        if any(len(p)!=2 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i<0 for i in p) or p[0]==p[1] for p in pairs):
            raise ValueError('valid atom pairs required')
        if len({tuple(sorted(p)) for p in pairs})!=len(pairs):raise ValueError('duplicate descriptor')

    def descriptor_jet(self, coordinates):
        v,J,H=squared_pair_distance_jet(coordinates,self.pairs)
        v-=self.reference_squared_distances;v[-1]/=self.active_scale
        J[-1]/=self.active_scale;H[-1]/=self.active_scale
        return v,J,H

    def descriptor_potential_jet(self, displacement):
        v=np.asarray(displacement,float);m=len(self.pairs)-1
        if v.shape!=(m+1,) or not np.all(np.isfinite(v)):raise ValueError('finite descriptor displacement required')
        z=v[:-1];t=v[-1]
        def u(k):return np.polynomial.polynomial.polyval(t,np.polynomial.polynomial.polyder(self.angular_coefficients,k))
        def a(k):return np.array([np.polynomial.polynomial.polyval(t,np.polynomial.polynomial.polyder(self.force_coefficients[:,i],k)) for i in range(m)])
        value=u(0)+a(0)@z+.5*z@self.passive_quadratic@z
        gradient=np.r_[a(0)+self.passive_quadratic@z,u(1)+a(1)@z]
        hessian=np.zeros((m+1,m+1));hessian[:m,:m]=self.passive_quadratic
        hessian[:m,m]=hessian[m,:m]=a(1);hessian[m,m]=u(2)+a(2)@z
        return float(value),gradient,hessian

    def cartesian_jet(self, coordinates):
        v,J,D=self.descriptor_jet(coordinates);value,g,H=self.descriptor_potential_jet(v)
        return value,J.T@g,J.T@H@J+np.einsum('i,ijk->jk',g,D)

    def to_dict(self):
        return dict(schema='matrix.architect.distance_angular_response.v1',pairs=[list(p) for p in self.pairs],
            reference_squared_distances=self.reference_squared_distances.tolist(),active_scale=float(self.active_scale),
            angular_coefficients=self.angular_coefficients.tolist(),force_coefficients=self.force_coefficients.tolist(),
            passive_quadratic=self.passive_quadratic.tolist(),scope='local finite-data angular/SAM response')

    @classmethod
    def from_dict(cls,data):
        if data.get('schema')!='matrix.architect.distance_angular_response.v1':raise ValueError('unsupported distance response')
        return cls(*(data[k] for k in ('pairs','reference_squared_distances','active_scale','angular_coefficients','force_coefficients','passive_quadratic')))


def fit_distance_angular_response(reference_coordinates, reference_gradient, reference_hessian,
        *, pairs, active_scale, reference_angular_first, training_coordinates,
        training_relative_energies, training_gradients, reference_angular_second=None):
    """Fit a local scalar response with a complete reference Cartesian two-jet.

    The active pair is last. Training geometries must be a frozen angular
    section: all passive pair distances equal their reference values. The
    caller supplies the active descriptor scale and its reference derivatives.
    All units are caller-consistent; no PES or validation callback is accepted.
    Full-rank nonplanar training gradients provide angular Hermite rows.
    """
    x=np.asarray(reference_coordinates,float);g=np.asarray(reference_gradient,float).reshape(-1)
    H=np.asarray(reference_hessian,float);xyz=np.asarray(training_coordinates,float)
    energies=np.asarray(training_relative_energies,float);gradients=np.asarray(training_gradients,float)
    r0,J0,D0=squared_pair_distance_jet(x,pairs);m=len(pairs)-1
    if (g.shape!=(x.size,) or H.shape!=(x.size,x.size) or xyz.ndim!=3
            or xyz.shape[1:]!=x.shape or energies.shape!=(len(xyz),)
            or gradients.shape!=(len(xyz),x.size) or not len(xyz)
            or not all(np.all(np.isfinite(v)) for v in (g,H,xyz,energies,gradients))
            or not np.allclose(H,H.T,rtol=0,atol=1e-7)
            or not np.isfinite(active_scale) or active_scale<=0):
        raise ValueError('finite compatible reference and frozen training jets required')
    def descriptor(y):
        v,J,D=squared_pair_distance_jet(y,pairs);v-=r0
        v[-1]/=active_scale;J[-1]/=active_scale;D[-1]/=active_scale
        return v,J,D
    rows=[];nodes=[]
    for y,e,gy in zip(xyz,energies,gradients):
        v,J,D=descriptor(y)
        if np.max(abs(v[:-1]))>1e-8:raise ValueError('training point changes passive distances')
        rows.append((v[-1],0,e));nodes.append(v[-1])
        singular=np.linalg.svd(J,compute_uv=False)
        if np.count_nonzero(singular>1e-10*singular[0])==len(pairs):
            derivative=np.linalg.lstsq(J.T,gy,rcond=1e-12)[0][-1]
            rows.append((v[-1],1,derivative))
    coeff=fit_anchored_scalar_polynomial(0.,reference_angular_first,rows,reference_second=reference_angular_second)
    def u(t,k):return np.polynomial.polynomial.polyval(t,np.polynomial.polynomial.polyder(coeff,k))
    forces=[];force_residuals=[]
    for y,gy in [(x,g),*zip(xyz,gradients)]:
        v,J,D=descriptor(y)
        if np.linalg.matrix_rank(J[:-1].T,tol=1e-10*np.linalg.norm(J[:-1],2))<m:raise ValueError('passive reference response is unobservable')
        target=gy-u(v[-1],1)*J[-1]
        f=np.linalg.lstsq(J[:-1].T,target,rcond=1e-12)[0]
        forces.append(f);force_residuals.append(float(np.max(abs(J[:-1].T@f-target))))
    vandermonde=np.vander(np.r_[0.,nodes],len(nodes)+1,increasing=True)
    if np.linalg.matrix_rank(vandermonde)<len(nodes)+1:raise ValueError('duplicate angular training descriptor')
    A=np.linalg.solve(vandermonde,np.array(forces))
    v,J,D=descriptor(x);first=np.r_[A[0],u(0,1)]
    h=np.zeros((m+1,m+1));h[:m,m]=h[m,:m]=A[1];h[m,m]=u(0,2)
    residual=H-np.einsum('i,ijk->jk',first,D)-J.T@h@J
    inverse=np.linalg.pinv(J[:-1]);B=inverse.T@residual@inverse;B=(B+B.T)/2
    model=DistanceAngularResponse(tuple(pairs),r0,active_scale,coeff,A,B)
    _,fg,fh=model.cartesian_jet(x)
    report=dict(training_points=len(xyz),angular_Hermite_observations=len(rows),
        angular_polynomial_degree=len(coeff)-1,force_polynomial_degree=len(A)-1,
        reference_gradient_max_residual=float(np.max(abs(fg-g))),
        reference_hessian_max_residual=float(np.max(abs(fh-H))),
        frozen_section_force_max_residuals=force_residuals,
        training_energy_max_residual=float(max(abs(model.cartesian_jet(y)[0]-e) for y,e in zip(xyz,energies))),
        reference_angular_second_supplied=reference_angular_second is not None,
        validation_observations_used=False)
    return model,report
