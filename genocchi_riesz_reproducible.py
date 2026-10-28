#!/usr/bin/env python3
"""Reproducible computations for the direct Genocchi--Riesz collocation paper.

Python 3.11+; dependencies: NumPy and SciPy.
The first N classical Genocchi polynomials span P_{N-1}. To avoid solving in
an extremely ill-conditioned raw Genocchi coefficient representation, this
script applies a QR-based change of coordinates to the sampled Genocchi basis.
The approximation space is unchanged. The Riesz operator is the one defined
in the manuscript: (D_{0+}^alpha + D_{1-}^alpha)/(2 cos(pi alpha/2)).

Outputs: results_errors.csv, results_diagnostics.csv, run_log.txt.
"""
import csv
import time
from math import comb, factorial
from pathlib import Path
import numpy as np
from scipy.special import gammaln, roots_jacobi, hyp2f1

OUT = Path(__file__).resolve().parent
ALPHAS = (1.2, 1.5, 1.8)
N_VALUES = {1: (4, 6, 8, 10, 12, 14, 16), 2: (4, 6, 8, 10, 12, 14, 16), 3: (4, 6, 8, 10, 12, 14), 4: (4, 6, 8, 10, 12, 14, 16)}
N_VALIDATION = 1024
SERIES_DEGREE = 100

def genocchi_numbers(nmax):
    G=np.zeros(nmax+1,dtype=np.longdouble)
    if nmax>=1: G[1]=1
    for n in range(2,nmax+1):
        s=np.longdouble(0)
        for k in range(1,n):
            s += np.longdouble(comb(n,k))*G[n-k]
        G[n]=-s/2
    return G

def genocchi_coeffs(N):
    gn=genocchi_numbers(N)
    C=np.zeros((N,N),dtype=float) # row j-1 coefficients x^k
    for j in range(1,N+1):
        for k in range(j):
            C[j-1,k]=float(comb(j,k)*gn[j-k])
    return C

def rl_left_coeffs(C, alpha, x):
    # C shape (N, degree+1), x vector strictly interior
    N,deg1=C.shape
    out=np.zeros((len(x),N))
    for j in range(N):
        for k,a in enumerate(C[j]):
            if a != 0:
                out[:,j] += a*np.exp(gammaln(k+1)-gammaln(k+1-alpha))*x**(k-alpha)
    return out

def rl_right_coeffs(C, alpha, x):
    N,deg1=C.shape
    out=np.zeros((len(x),N))
    # D^alpha_{x,1} x^k = sum_r (-1)^r C(k,r) Gamma(r+1)/Gamma(r+1-alpha)*(1-x)^(r-alpha)
    for j in range(N):
        for k,a in enumerate(C[j]):
            if a == 0: continue
            for r in range(k+1):
                out[:,j] += a*((-1)**r)*comb(k,r)*np.exp(gammaln(r+1)-gammaln(r+1-alpha))*(1-x)**(r-alpha)
    return out

def riesz_matrix(C, alpha, x):
    return (rl_left_coeffs(C,alpha,x)+rl_right_coeffs(C,alpha,x))/(2*np.cos(np.pi*alpha/2))

def polyval_rows(C,x):
    # C N x K -> len(x) x N
    return np.polynomial.polynomial.polyval(x[:,None], C.T).T if False else np.column_stack([np.polynomial.polynomial.polyval(x,c) for c in C])

def gl_weighted(alpha,n=160):
    # Gauss-Jacobi for z^(1-alpha), z in [0,1]
    beta=1-alpha
    y,w=roots_jacobi(n,0,beta)
    z=(y+1)/2
    w=w/(2**(beta+1))
    return z,w

def caputo_left_from_second(u2,x,alpha,nq=160):
    z,w=gl_weighted(alpha,nq)
    # x^(2-a) int_0^1 z^(1-a) u''(x*(1-z)) dz
    vals=np.array([np.sum(w*u2(xx*(1-z))) * xx**(2-alpha) / np.exp(gammaln(2-alpha)) for xx in np.atleast_1d(x)])
    return vals

def caputo_right_from_second(u2,x,alpha,nq=160):
    z,w=gl_weighted(alpha,nq)
    vals=np.array([np.sum(w*u2(xx+(1-xx)*z)) * (1-xx)**(2-alpha) / np.exp(gammaln(2-alpha)) for xx in np.atleast_1d(x)])
    return vals

def source_exact1(x,alpha):
    # u=x^3(1-x)^3 = x^3 - 3x^4 +3x^5 - x^6
    C=np.array([[0,0,0,1,-3,3,-1]],float)
    return riesz_matrix(C,alpha,np.asarray(x))[:,0]

def u2_ex2(x):
    return 2*np.pi*np.cos(np.pi*x*x)-4*np.pi**2*x*x*np.sin(np.pi*x*x)
def source2_right_coeff(M):
    """Taylor coefficients of sin(pi*(1-y)^2)=sin(2*pi*y-pi*y^2)."""
    z=np.zeros(M+1); z[1]=2*np.pi; z[2]=-np.pi
    out=np.zeros(M+1); power=np.zeros(M+1); power[0]=1.0
    for n in range(1,M+1):
        power=np.convolve(power,z)[:M+1]
        if n % 2 == 1:
            out += ((-1)**((n-1)//2)/factorial(n))*power
    return out

def source_exact2(x,alpha,nq=160):
    """Analytical endpoint power-series evaluation of R_alpha sin(pi*x^2)."""
    x=np.asarray(x,dtype=float)
    M=max(80,nq)
    left=np.zeros_like(x)
    # sin(pi*x^2) = sum_m (-1)^m*pi^(2m+1)*x^(4m+2)/(2m+1)!
    for m in range(M//4+1):
        k=4*m+2
        if k>M: break
        a=(-1)**m*np.pi**(2*m+1)/factorial(2*m+1)
        left += a*np.exp(gammaln(k+1)-gammaln(k+1-alpha))*x**(k-alpha)
    # For the right RL derivative, expand u(1-y) in y=1-x.
    rc=source2_right_coeff(M)
    y=1-x; right=np.zeros_like(x)
    for k,a in enumerate(rc):
        if a:
            right += a*np.exp(gammaln(k+1)-gammaln(k+1-alpha))*y**(k-alpha)
    return (left+right)/(2*np.cos(np.pi*alpha/2))

def source_exact3(x,alpha,M=100):
    # u=x(1-x)e^x = x e^x - x^2 e^x; Taylor coeffs a_k = 1/(k-1)! - 1/(k-2)! for k>=1 (for k=1, second term absent)
    # approximate by degree M polynomial, then exact RL derivative of that polynomial
    coeff=np.zeros(M+1)
    for k in range(1,M+1):
        coeff[k]=1/factorial(k-1) if False else 0
    # stable recurrence for factorial terms
    for k in range(1,M+1):
        a1=1/factorial(k-1)
        a2=1/factorial(k-2) if k>=2 else 0
        coeff[k]=a1-a2
    C=coeff.reshape(1,-1)
    return riesz_matrix(C,alpha,np.asarray(x))[:,0]

def source_exact4(x,alpha,beta=1.8):
    """Riesz source for u=x^beta(1-x)^beta using a hypergeometric formula.

    For u(x)=u(1-x), the right derivative at x equals the left
    derivative evaluated at 1-x, not generally at x itself. Thus
    R_alpha u=(D_left(x)+D_left(1-x))/(2 cos(pi alpha/2)).
    """
    x=np.asarray(x,dtype=float)
    pref=np.exp(gammaln(beta+1)-gammaln(beta+1-alpha))
    dleft_x=pref*x**(beta-alpha)*hyp2f1(-beta,beta+1,beta+1-alpha,x)
    dleft_ref=pref*(1-x)**(beta-alpha)*hyp2f1(-beta,beta+1,beta+1-alpha,1-x)
    return (dleft_x+dleft_ref)/(2*np.cos(np.pi*alpha/2))

def solve_example(N,alpha,example,sourceM=SERIES_DEGREE):
    """Solve one problem in QR-reparameterized Genocchi coordinates."""
    C=genocchi_coeffs(N)
    x=(1-np.cos(np.pi*np.arange(N)/(N-1)))/2
    V=polyval_rows(C,x)
    # V = Q R. New basis functions are linear combinations G R^{-1}; same space.
    Q,R=np.linalg.qr(V)
    T=np.linalg.solve(R,np.eye(N))
    A=np.zeros((N,N)); b=np.zeros(N)
    A[0,:]=V[0,:]; A[-1,:]=V[-1,:]
    A[1:-1,:]=riesz_matrix(C,alpha,x[1:-1])
    if example==1:
        b[1:-1]=source_exact1(x[1:-1],alpha)
        uex=lambda xx: xx**3*(1-xx)**3
    elif example==2:
        b[1:-1]=source_exact2(x[1:-1],alpha)
        uex=lambda xx: np.sin(np.pi*xx**2)
    elif example==3:
        b[1:-1]=source_exact3(x[1:-1],alpha,sourceM)
        uex=lambda xx: xx*(1-xx)*np.exp(xx)
    else:
        if abs(alpha-1.5)>1e-12:
            raise ValueError("Example 4 is run at alpha=1.5")
        b[1:-1]=source_exact4(x[1:-1],alpha)
        uex=lambda xx: xx**1.8*(1-xx)**1.8
    An=A@T
    # Column scaling is retained as an additional inexpensive safeguard.
    colnorm=np.linalg.norm(An,axis=0)
    As=An/colnorm
    z=np.linalg.solve(As,b)
    d=z/colnorm
    c=T@d
    xt=np.linspace(0,1,N_VALIDATION)
    Ptest=polyval_rows(C,xt)
    ut=Ptest@c
    err=float(np.max(np.abs(uex(xt)-ut)))
    # Direct solve in the original Genocchi coordinates, used only as a
    # diagnostic comparison with the QR-reparameterized solve.
    c_raw=np.linalg.solve(A,b)
    raw_err=float(np.max(np.abs(uex(xt)-Ptest@c_raw)))
    algebraic=float(np.linalg.norm(An@d-b,np.inf))
    # Independent residual grid; avoid endpoint singular evaluation.
    xv=np.linspace(.02,.98,101)
    if example==1: rhs=source_exact1(xv,alpha)
    elif example==2: rhs=source_exact2(xv,alpha)
    elif example==3: rhs=source_exact3(xv,alpha,sourceM)
    else: rhs=source_exact4(xv,alpha)
    residual=float(np.max(np.abs(riesz_matrix(C,alpha,xv)@c-rhs)))
    return {"error":err,"raw_error":raw_err,"error_difference":abs(err-raw_err),"cond_raw":float(np.linalg.cond(A)),
            "cond_qr":float(np.linalg.cond(An)),
            "cond_qr_scaled":float(np.linalg.cond(As)),
            "algebraic_residual":algebraic,"independent_residual":residual}

def symmetry_check():
    """Check R_alpha G_n(1-x)=(-1)^(n-1) R_alpha G_n(x)."""
    C=genocchi_coeffs(14); x=np.linspace(.01,.99,99)
    rows=[]
    for alpha in ALPHAS:
        left=riesz_matrix(C,alpha,x)
        reflected=riesz_matrix(C,alpha,1-x)
        parity=np.array([(-1)**j for j in range(C.shape[0])])
        absolute=float(np.max(np.abs(reflected-left*parity)))
        relative=absolute/max(1.0,float(np.max(np.abs(left))))
        rows.append((alpha,absolute,relative))
    return rows

def main():
    t0=time.perf_counter()
    error_rows=[]; diag_rows=[]; comparison_rows=[]
    lines=[]
    lines.append("Direct Genocchi-Riesz collocation reproducibility run")
    lines.append(f"NumPy {np.__version__}; validation points={N_VALIDATION}; Taylor degree={SERIES_DEGREE}")
    for ex, Ns in N_VALUES.items():
        lines.append(f"\nExample {ex}")
        for N in Ns:
            vals=[]
            alphas=(1.5,) if ex==4 else ALPHAS
            for alpha in alphas:
                r=solve_example(N,alpha,ex)
                error_rows.append([ex,N,alpha,r['error']])
                comparison_rows.append([ex,N,alpha,r['raw_error'],r['error'],r['error_difference']])
                diag_rows.append([ex,N,alpha,r['cond_raw'],r['cond_qr'],r['cond_qr_scaled'],r['algebraic_residual'],r['independent_residual']])
                vals.append(f"a={alpha:g}: E_QR={r['error']:.4e}, E_raw={r['raw_error']:.4e}, cond(QR)={r['cond_qr']:.3e}, alg={r['algebraic_residual']:.2e}, residual={r['independent_residual']:.2e}")
            lines.append(f"N={N}: " + " | ".join(vals))
    lines.append("\nReflection identity check")
    for alpha,absolute,relative in symmetry_check():
        lines.append(f"alpha={alpha:g}: abs={absolute:.4e}, rel={relative:.4e}")
    lines.append(f"\nElapsed seconds: {time.perf_counter()-t0:.3f}")
    with open(OUT/'results_errors.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['example','N','alpha','E_inf_1024']); w.writerows(error_rows)
    with open(OUT/'results_diagnostics.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['example','N','alpha','cond_raw','cond_qr_transformed','cond_qr_column_scaled','algebraic_residual_inf','independent_residual_inf_x_0.02_to_0.98']); w.writerows(diag_rows)
    with open(OUT/'results_basis_comparison.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['example','N','alpha','E_inf_raw_coordinates','E_inf_QR_coordinates','absolute_error_difference']); w.writerows(comparison_rows)
    (OUT/'run_log.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print(f"\nWrote: {OUT/'results_errors.csv'}, {OUT/'results_diagnostics.csv'}, {OUT/'results_basis_comparison.csv'}, {OUT/'run_log.txt'}")

if __name__=='__main__':
    main()
