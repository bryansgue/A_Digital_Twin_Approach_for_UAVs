/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Drone_ode_constr_h_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_house CASADI_PREFIX(house)
#define casadi_if_else CASADI_PREFIX(if_else)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_qr CASADI_PREFIX(qr)
#define casadi_qr_colcomb CASADI_PREFIX(qr_colcomb)
#define casadi_qr_mv CASADI_PREFIX(qr_mv)
#define casadi_qr_singular CASADI_PREFIX(qr_singular)
#define casadi_qr_solve CASADI_PREFIX(qr_solve)
#define casadi_qr_trs CASADI_PREFIX(qr_trs)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_scal CASADI_PREFIX(scal)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_if_else(casadi_real c, casadi_real x, casadi_real y) { return c!=0 ? x : y;}

void casadi_scal(casadi_int n, casadi_real alpha, casadi_real* x) {
  casadi_int i;
  if (!x) return;
  for (i=0; i<n; ++i) *x++ *= alpha;
}

casadi_real casadi_house(casadi_real* v, casadi_real* beta, casadi_int nv) {
  casadi_int i;
  casadi_real v0, sigma, s, sigma_is_zero, v0_nonpos;
  v0 = v[0];
  sigma=0;
  for (i=1; i<nv; ++i) sigma += v[i]*v[i];
  s = sqrt(v0*v0 + sigma);
  sigma_is_zero = sigma==0;
  v0_nonpos = v0<=0;
  v[0] = casadi_if_else(sigma_is_zero, 1,
                 casadi_if_else(v0_nonpos, v0-s, -sigma/(v0+s)));
  *beta = casadi_if_else(sigma_is_zero, 2*v0_nonpos, -1/(s*v[0]));
  return s;
}
void casadi_qr(const casadi_int* sp_a, const casadi_real* nz_a, casadi_real* x,
               const casadi_int* sp_v, casadi_real* nz_v, const casadi_int* sp_r, casadi_real* nz_r, casadi_real* beta,
               const casadi_int* prinv, const casadi_int* pc) {
   casadi_int ncol, nrow, r, c, k, k1;
   casadi_real alpha;
   const casadi_int *a_colind, *a_row, *v_colind, *v_row, *r_colind, *r_row;
   ncol = sp_a[1];
   a_colind=sp_a+2; a_row=sp_a+2+ncol+1;
   nrow = sp_v[0];
   v_colind=sp_v+2; v_row=sp_v+2+ncol+1;
   r_colind=sp_r+2; r_row=sp_r+2+ncol+1;
   for (r=0; r<nrow; ++r) x[r] = 0;
   for (c=0; c<ncol; ++c) {
     for (k=a_colind[pc[c]]; k<a_colind[pc[c]+1]; ++k) x[prinv[a_row[k]]] = nz_a[k];
     for (k=r_colind[c]; k<r_colind[c+1] && (r=r_row[k])<c; ++k) {
       alpha = 0;
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) alpha += nz_v[k1]*x[v_row[k1]];
       alpha *= beta[r];
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) x[v_row[k1]] -= alpha*nz_v[k1];
       *nz_r++ = x[r];
       x[r] = 0;
     }
     for (k=v_colind[c]; k<v_colind[c+1]; ++k) {
       nz_v[k] = x[v_row[k]];
       x[v_row[k]] = 0;
     }
     *nz_r++ = casadi_house(nz_v + v_colind[c], beta + c, v_colind[c+1] - v_colind[c]);
   }
 }
void casadi_qr_mv(const casadi_int* sp_v, const casadi_real* v, const casadi_real* beta, casadi_real* x,
                  casadi_int tr) {
  casadi_int ncol, c, c1, k;
  casadi_real alpha;
  const casadi_int *colind, *row;
  ncol=sp_v[1];
  colind=sp_v+2; row=sp_v+2+ncol+1;
  for (c1=0; c1<ncol; ++c1) {
    c = tr ? c1 : ncol-1-c1;
    alpha=0;
    for (k=colind[c]; k<colind[c+1]; ++k) alpha += v[k]*x[row[k]];
    alpha *= beta[c];
    for (k=colind[c]; k<colind[c+1]; ++k) x[row[k]] -= alpha*v[k];
  }
}
void casadi_qr_trs(const casadi_int* sp_r, const casadi_real* nz_r, casadi_real* x, casadi_int tr) {
  casadi_int ncol, r, c, k;
  const casadi_int *colind, *row;
  ncol=sp_r[1];
  colind=sp_r+2; row=sp_r+2+ncol+1;
  if (tr) {
    for (c=0; c<ncol; ++c) {
      for (k=colind[c]; k<colind[c+1]; ++k) {
        r = row[k];
        if (r==c) {
          x[c] /= nz_r[k];
        } else {
          x[c] -= nz_r[k]*x[r];
        }
      }
    }
  } else {
    for (c=ncol-1; c>=0; --c) {
      for (k=colind[c+1]-1; k>=colind[c]; --k) {
        r=row[k];
        if (r==c) {
          x[r] /= nz_r[k];
        } else {
          x[r] -= nz_r[k]*x[c];
        }
      }
    }
  }
}
void casadi_qr_solve(casadi_real* x, casadi_int nrhs, casadi_int tr,
                     const casadi_int* sp_v, const casadi_real* v, const casadi_int* sp_r, const casadi_real* r,
                     const casadi_real* beta, const casadi_int* prinv, const casadi_int* pc, casadi_real* w) {
  casadi_int k, c, nrow_ext, ncol;
  nrow_ext = sp_v[0]; ncol = sp_v[1];
  for (k=0; k<nrhs; ++k) {
    if (tr) {
      for (c=0; c<ncol; ++c) w[c] = x[pc[c]];
      casadi_qr_trs(sp_r, r, w, 1);
      casadi_qr_mv(sp_v, v, beta, w, 0);
      for (c=0; c<ncol; ++c) x[c] = w[prinv[c]];
    } else {
      for (c=0; c<nrow_ext; ++c) w[c] = 0;
      for (c=0; c<ncol; ++c) w[prinv[c]] = x[c];
      casadi_qr_mv(sp_v, v, beta, w, 1);
      casadi_qr_trs(sp_r, r, w, 0);
      for (c=0; c<ncol; ++c) x[pc[c]] = w[c];
    }
    x += ncol;
  }
}
casadi_int casadi_qr_singular(casadi_real* rmin, casadi_int* irmin, const casadi_real* nz_r,
                             const casadi_int* sp_r, const casadi_int* pc, casadi_real eps) {
  casadi_real rd, rd_min;
  casadi_int ncol, c, nullity;
  const casadi_int* r_colind;
  nullity = 0;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  for (c=0; c<ncol; ++c) {
    rd = fabs(nz_r[r_colind[c+1]-1]);
    if (rd<eps) nullity++;
    if (c==0 || rd < rd_min) {
      rd_min = rd;
      if (rmin) *rmin = rd;
      if (irmin) *irmin = pc[c];
    }
  }
  return nullity;
}
void casadi_qr_colcomb(casadi_real* v, const casadi_real* nz_r, const casadi_int* sp_r,
                       const casadi_int* pc, casadi_real eps, casadi_int ind) {
  casadi_int ncol, r, c, k;
  const casadi_int *r_colind, *r_row;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  r_row = r_colind + ncol + 1;
  for (c=0; c<ncol; ++c) {
    if (fabs(nz_r[r_colind[c+1]-1])<eps && 0==ind--) {
      ind = c;
      break;
    }
  }
  casadi_clear(v, ncol);
  v[pc[ind]] = 1.;
  for (k=r_colind[ind]; k<r_colind[ind+1]-1; ++k) {
    v[pc[r_row[k]]] = -nz_r[k];
  }
  for (c=ind-1; c>=0; --c) {
    for (k=r_colind[c+1]-1; k>=r_colind[c]; --k) {
      r=r_row[k];
      if (r==c) {
        if (fabs(nz_r[k])<eps) {
          v[pc[r]] = 0;
        } else {
          v[pc[r]] /= nz_r[k];
        }
      } else {
        v[pc[r]] -= nz_r[k]*v[pc[c]];
      }
    }
  }
  casadi_scal(ncol, 1./sqrt(casadi_dot(ncol, v, v)), v);
}

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

static const casadi_int casadi_s0[4] = {0, 1, 2, 3};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s2[17] = {4, 4, 0, 4, 7, 9, 10, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
static const casadi_int casadi_s3[17] = {4, 4, 0, 1, 3, 6, 10, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s6[16] = {1, 11, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0};
static const casadi_int casadi_s7[11] = {1, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0};
static const casadi_int casadi_s8[51] = {11, 4, 0, 11, 22, 33, 44, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s9[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s10[3] = {0, 0, 0};
static const casadi_int casadi_s11[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
static const casadi_real casadi_c1[16] = {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};

/* Drone_ode_constr_h_fun:(i0[11],i1[4],i2[],i3[17])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+12, w2, w3, w4, *w5=w+17, w6, w7, w8, w9, *w10=w+31, *w11=w+42, *w12=w+63, *w13=w+72, *w14=w+81, *w15=w+90, w16, *w17=w+95, *w18=w+104, *w19=w+107, *w20=w+140, *w21=w+173, w22, w23, w24, w25, w26, *w27=w+190, *w28=w+194, *w29=w+198, *w30=w+202, *w31=w+218, *w32=w+234, *w33=w+278, *w34=w+322, *w35=w+350, *w36=w+366, *w37=w+382, *w38=w+426, *w39=w+547, *w40=w+668;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(11x1,2nz) */
  casadi_clear(w1, 2);
  /* #2: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #3: @3 = input[3][15] */
  w3 = arg[3] ? arg[3][15] : 0;
  /* #4: @3 = (@2-@3) */
  w3  = (w2-w3);
  /* #5: @4 = (2.*@3) */
  w4 = (2.* w3 );
  /* #6: @5 = ones(11x1,10nz) */
  casadi_fill(w5, 10, 1.);
  /* #7: {@6, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@5) */
  w6 = w5[0];
  /* #8: @4 = (@4*@6) */
  w4 *= w6;
  /* #9: @3 = sq(@3) */
  w3 = casadi_sq( w3 );
  /* #10: @6 = input[0][1] */
  w6 = arg[0] ? arg[0][1] : 0;
  /* #11: @7 = input[3][16] */
  w7 = arg[3] ? arg[3][16] : 0;
  /* #12: @7 = (@6-@7) */
  w7  = (w6-w7);
  /* #13: @8 = sq(@7) */
  w8 = casadi_sq( w7 );
  /* #14: @3 = (@3+@8) */
  w3 += w8;
  /* #15: @3 = sqrt(@3) */
  w3 = sqrt( w3 );
  /* #16: @8 = (2.*@3) */
  w8 = (2.* w3 );
  /* #17: @4 = (@4/@8) */
  w4 /= w8;
  /* #18: @4 = (-@4) */
  w4 = (- w4 );
  /* #19: (@1[0] = @4) */
  for (rr=w1+0, ss=(&w4); rr!=w1+1; rr+=1) *rr = *ss++;
  /* #20: @7 = (2.*@7) */
  w7 = (2.* w7 );
  /* #21: @4 = ones(11x1,1nz) */
  w4 = 1.;
  /* #22: {NULL, @9, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@4) */
  w9 = w4;
  /* #23: @7 = (@7*@9) */
  w7 *= w9;
  /* #24: @7 = (@7/@8) */
  w7 /= w8;
  /* #25: @7 = (-@7) */
  w7 = (- w7 );
  /* #26: (@1[1] = @7) */
  for (rr=w1+1, ss=(&w7); rr!=w1+2; rr+=1) *rr = *ss++;
  /* #27: @1 = @1' */
  /* #28: @10 = zeros(11x1) */
  casadi_clear(w10, 11);
  /* #29: @11 = zeros(3x7) */
  casadi_clear(w11, 21);
  /* #30: @12 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w12);
  /* #31: @13 = zeros(3x3) */
  casadi_clear(w13, 9);
  /* #32: @14 = zeros(3x3) */
  casadi_clear(w14, 9);
  /* #33: @7 = input[0][3] */
  w7 = arg[0] ? arg[0][3] : 0;
  /* #34: @8 = input[0][4] */
  w8 = arg[0] ? arg[0][4] : 0;
  /* #35: @9 = input[0][5] */
  w9 = arg[0] ? arg[0][5] : 0;
  /* #36: @4 = input[0][6] */
  w4 = arg[0] ? arg[0][6] : 0;
  /* #37: @15 = vertcat(@7, @8, @9, @4) */
  rr=w15;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w4;
  /* #38: @16 = ||@15||_F */
  w16 = sqrt(casadi_dot(4, w15, w15));
  /* #39: @15 = (@15/@16) */
  for (i=0, rr=w15; i<4; ++i) (*rr++) /= w16;
  /* #40: @16 = @15[3] */
  for (rr=(&w16), ss=w15+3; ss!=w15+4; ss+=1) *rr++ = *ss;
  /* #41: @16 = (-@16) */
  w16 = (- w16 );
  /* #42: (@14[3] = @16) */
  for (rr=w14+3, ss=(&w16); rr!=w14+4; rr+=1) *rr = *ss++;
  /* #43: @16 = @15[2] */
  for (rr=(&w16), ss=w15+2; ss!=w15+3; ss+=1) *rr++ = *ss;
  /* #44: (@14[6] = @16) */
  for (rr=w14+6, ss=(&w16); rr!=w14+7; rr+=1) *rr = *ss++;
  /* #45: @16 = @15[1] */
  for (rr=(&w16), ss=w15+1; ss!=w15+2; ss+=1) *rr++ = *ss;
  /* #46: @16 = (-@16) */
  w16 = (- w16 );
  /* #47: (@14[7] = @16) */
  for (rr=w14+7, ss=(&w16); rr!=w14+8; rr+=1) *rr = *ss++;
  /* #48: @16 = @15[3] */
  for (rr=(&w16), ss=w15+3; ss!=w15+4; ss+=1) *rr++ = *ss;
  /* #49: (@14[1] = @16) */
  for (rr=w14+1, ss=(&w16); rr!=w14+2; rr+=1) *rr = *ss++;
  /* #50: @16 = @15[2] */
  for (rr=(&w16), ss=w15+2; ss!=w15+3; ss+=1) *rr++ = *ss;
  /* #51: @16 = (-@16) */
  w16 = (- w16 );
  /* #52: (@14[2] = @16) */
  for (rr=w14+2, ss=(&w16); rr!=w14+3; rr+=1) *rr = *ss++;
  /* #53: @16 = @15[1] */
  for (rr=(&w16), ss=w15+1; ss!=w15+2; ss+=1) *rr++ = *ss;
  /* #54: (@14[5] = @16) */
  for (rr=w14+5, ss=(&w16); rr!=w14+6; rr+=1) *rr = *ss++;
  /* #55: @17 = (2.*@14) */
  for (i=0, rr=w17, cs=w14; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #56: @13 = mac(@17,@14,@13) */
  for (i=0, rr=w13; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w17+j, tt=w14+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #57: @12 = (@12+@13) */
  for (i=0, rr=w12, cs=w13; i<9; ++i) (*rr++) += (*cs++);
  /* #58: @16 = @15[0] */
  for (rr=(&w16), ss=w15+0; ss!=w15+1; ss+=1) *rr++ = *ss;
  /* #59: @16 = (2.*@16) */
  w16 = (2.* w16 );
  /* #60: @14 = (@16*@14) */
  for (i=0, rr=w14, cs=w14; i<9; ++i) (*rr++)  = (w16*(*cs++));
  /* #61: @12 = (@12+@14) */
  for (i=0, rr=w12, cs=w14; i<9; ++i) (*rr++) += (*cs++);
  /* #62: @18 = zeros(3x1) */
  casadi_clear(w18, 3);
  /* #63: @19 = horzcat(@11, @12, @18) */
  rr=w19;
  for (i=0, cs=w11; i<21; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<9; ++i) *rr++ = *cs++;
  for (i=0, cs=w18; i<3; ++i) *rr++ = *cs++;
  /* #64: @20 = @19' */
  for (i=0, rr=w20, cs=w19; i<11; ++i) for (j=0; j<3; ++j) rr[i+j*11] = *cs++;
  /* #65: @21 = zeros(4x3) */
  casadi_clear(w21, 12);
  /* #66: @16 = 0.5 */
  w16 = 5.0000000000000000e-01;
  /* #67: @22 = 0 */
  w22 = 0.;
  /* #68: @23 = 0 */
  w23 = 0.;
  /* #69: @24 = 0 */
  w24 = 0.;
  /* #70: @25 = input[0][10] */
  w25 = arg[0] ? arg[0][10] : 0;
  /* #71: @26 = (-@25) */
  w26 = (- w25 );
  /* #72: @15 = horzcat(@22, @23, @24, @26) */
  rr=w15;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w24;
  *rr++ = w26;
  /* #73: @15 = @15' */
  /* #74: @22 = 0 */
  w22 = 0.;
  /* #75: @23 = 0 */
  w23 = 0.;
  /* #76: @24 = 0 */
  w24 = 0.;
  /* #77: @27 = horzcat(@22, @23, @25, @24) */
  rr=w27;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w25;
  *rr++ = w24;
  /* #78: @27 = @27' */
  /* #79: @22 = 0 */
  w22 = 0.;
  /* #80: @23 = (-@25) */
  w23 = (- w25 );
  /* #81: @24 = 0 */
  w24 = 0.;
  /* #82: @26 = 0 */
  w26 = 0.;
  /* #83: @28 = horzcat(@22, @23, @24, @26) */
  rr=w28;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w24;
  *rr++ = w26;
  /* #84: @28 = @28' */
  /* #85: @22 = 0 */
  w22 = 0.;
  /* #86: @23 = 0 */
  w23 = 0.;
  /* #87: @24 = 0 */
  w24 = 0.;
  /* #88: @29 = horzcat(@25, @22, @23, @24) */
  rr=w29;
  *rr++ = w25;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w24;
  /* #89: @29 = @29' */
  /* #90: @30 = horzcat(@15, @27, @28, @29) */
  rr=w30;
  for (i=0, cs=w15; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w27; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w28; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w29; i<4; ++i) *rr++ = *cs++;
  /* #91: @31 = @30' */
  for (i=0, rr=w31, cs=w30; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #92: @31 = (@16*@31) */
  for (i=0, rr=w31, cs=w31; i<16; ++i) (*rr++)  = (w16*(*cs++));
  /* #93: @30 = zeros(4x4) */
  casadi_clear(w30, 16);
  /* #94: @32 = horzcat(@21, @31, @30) */
  rr=w32;
  for (i=0, cs=w21; i<12; ++i) *rr++ = *cs++;
  for (i=0, cs=w31; i<16; ++i) *rr++ = *cs++;
  for (i=0, cs=w30; i<16; ++i) *rr++ = *cs++;
  /* #95: @33 = @32' */
  for (i=0, rr=w33, cs=w32; i<11; ++i) for (j=0; j<4; ++j) rr[i+j*11] = *cs++;
  /* #96: @34 = zeros(4x7) */
  casadi_clear(w34, 28);
  /* #97: @31 = zeros(4x4) */
  casadi_clear(w31, 16);
  /* #98: @30 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c1, 16, w30);
  /* #99: @35 = zeros(4x4) */
  casadi_clear(w35, 16);
  /* #100: @16 = 0.6756 */
  w16 = 6.7559999999999998e-01;
  /* #101: (@35[0] = @16) */
  for (rr=w35+0, ss=(&w16); rr!=w35+1; rr+=1) *rr = *ss++;
  /* #102: @16 = 0 */
  w16 = 0.;
  /* #103: (@35[4] = @16) */
  for (rr=w35+4, ss=(&w16); rr!=w35+5; rr+=1) *rr = *ss++;
  /* #104: @16 = 0 */
  w16 = 0.;
  /* #105: (@35[8] = @16) */
  for (rr=w35+8, ss=(&w16); rr!=w35+9; rr+=1) *rr = *ss++;
  /* #106: @16 = 0 */
  w16 = 0.;
  /* #107: (@35[12] = @16) */
  for (rr=w35+12, ss=(&w16); rr!=w35+13; rr+=1) *rr = *ss++;
  /* #108: @16 = 0 */
  w16 = 0.;
  /* #109: (@35[1] = @16) */
  for (rr=w35+1, ss=(&w16); rr!=w35+2; rr+=1) *rr = *ss++;
  /* #110: @16 = 0.6344 */
  w16 = 6.3439999999999996e-01;
  /* #111: (@35[5] = @16) */
  for (rr=w35+5, ss=(&w16); rr!=w35+6; rr+=1) *rr = *ss++;
  /* #112: @16 = 0 */
  w16 = 0.;
  /* #113: (@35[9] = @16) */
  for (rr=w35+9, ss=(&w16); rr!=w35+10; rr+=1) *rr = *ss++;
  /* #114: @16 = 0 */
  w16 = 0.;
  /* #115: (@35[13] = @16) */
  for (rr=w35+13, ss=(&w16); rr!=w35+14; rr+=1) *rr = *ss++;
  /* #116: @16 = 0 */
  w16 = 0.;
  /* #117: (@35[2] = @16) */
  for (rr=w35+2, ss=(&w16); rr!=w35+3; rr+=1) *rr = *ss++;
  /* #118: @16 = 0 */
  w16 = 0.;
  /* #119: (@35[6] = @16) */
  for (rr=w35+6, ss=(&w16); rr!=w35+7; rr+=1) *rr = *ss++;
  /* #120: @16 = 0.408 */
  w16 = 4.0799999999999997e-01;
  /* #121: (@35[10] = @16) */
  for (rr=w35+10, ss=(&w16); rr!=w35+11; rr+=1) *rr = *ss++;
  /* #122: @16 = 0 */
  w16 = 0.;
  /* #123: (@35[14] = @16) */
  for (rr=w35+14, ss=(&w16); rr!=w35+15; rr+=1) *rr = *ss++;
  /* #124: @16 = 0 */
  w16 = 0.;
  /* #125: (@35[3] = @16) */
  for (rr=w35+3, ss=(&w16); rr!=w35+4; rr+=1) *rr = *ss++;
  /* #126: @16 = 0 */
  w16 = 0.;
  /* #127: (@35[7] = @16) */
  for (rr=w35+7, ss=(&w16); rr!=w35+8; rr+=1) *rr = *ss++;
  /* #128: @16 = 0 */
  w16 = 0.;
  /* #129: (@35[11] = @16) */
  for (rr=w35+11, ss=(&w16); rr!=w35+12; rr+=1) *rr = *ss++;
  /* #130: @16 = 0.2953 */
  w16 = 2.9530000000000001e-01;
  /* #131: (@35[15] = @16) */
  for (rr=w35+15, ss=(&w16); rr!=w35+16; rr+=1) *rr = *ss++;
  /* #132: @30 = (@35\@30) */
  rr = w30;
  ss = w35;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[10], r[10], beta[4], w[8];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 4, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #133: @36 = zeros(4x4) */
  casadi_clear(w36, 16);
  /* #134: @16 = 0.5941 */
  w16 = 5.9409999999999996e-01;
  /* #135: (@36[0] = @16) */
  for (rr=w36+0, ss=(&w16); rr!=w36+1; rr+=1) *rr = *ss++;
  /* #136: @16 = -0.8109 */
  w16 = -8.1089999999999995e-01;
  /* #137: @16 = (@16*@25) */
  w16 *= w25;
  /* #138: (@36[4] = @16) */
  for (rr=w36+4, ss=(&w16); rr!=w36+5; rr+=1) *rr = *ss++;
  /* #139: @16 = 0 */
  w16 = 0.;
  /* #140: (@36[8] = @16) */
  for (rr=w36+8, ss=(&w16); rr!=w36+9; rr+=1) *rr = *ss++;
  /* #141: @16 = 0 */
  w16 = 0.;
  /* #142: (@36[12] = @16) */
  for (rr=w36+12, ss=(&w16); rr!=w36+13; rr+=1) *rr = *ss++;
  /* #143: @16 = 0.3984 */
  w16 = 3.9839999999999998e-01;
  /* #144: @16 = (@16*@25) */
  w16 *= w25;
  /* #145: (@36[1] = @16) */
  for (rr=w36+1, ss=(&w16); rr!=w36+2; rr+=1) *rr = *ss++;
  /* #146: @16 = 0.704 */
  w16 = 7.0399999999999996e-01;
  /* #147: (@36[5] = @16) */
  for (rr=w36+5, ss=(&w16); rr!=w36+6; rr+=1) *rr = *ss++;
  /* #148: @16 = 0 */
  w16 = 0.;
  /* #149: (@36[9] = @16) */
  for (rr=w36+9, ss=(&w16); rr!=w36+10; rr+=1) *rr = *ss++;
  /* #150: @16 = 0 */
  w16 = 0.;
  /* #151: (@36[13] = @16) */
  for (rr=w36+13, ss=(&w16); rr!=w36+14; rr+=1) *rr = *ss++;
  /* #152: @16 = 0 */
  w16 = 0.;
  /* #153: (@36[2] = @16) */
  for (rr=w36+2, ss=(&w16); rr!=w36+3; rr+=1) *rr = *ss++;
  /* #154: @16 = 0 */
  w16 = 0.;
  /* #155: (@36[6] = @16) */
  for (rr=w36+6, ss=(&w16); rr!=w36+7; rr+=1) *rr = *ss++;
  /* #156: @16 = 0.9365 */
  w16 = 9.3650000000000000e-01;
  /* #157: (@36[10] = @16) */
  for (rr=w36+10, ss=(&w16); rr!=w36+11; rr+=1) *rr = *ss++;
  /* #158: @16 = 0 */
  w16 = 0.;
  /* #159: (@36[14] = @16) */
  for (rr=w36+14, ss=(&w16); rr!=w36+15; rr+=1) *rr = *ss++;
  /* #160: @16 = 0 */
  w16 = 0.;
  /* #161: (@36[3] = @16) */
  for (rr=w36+3, ss=(&w16); rr!=w36+4; rr+=1) *rr = *ss++;
  /* #162: @16 = 0 */
  w16 = 0.;
  /* #163: (@36[7] = @16) */
  for (rr=w36+7, ss=(&w16); rr!=w36+8; rr+=1) *rr = *ss++;
  /* #164: @16 = 0 */
  w16 = 0.;
  /* #165: (@36[11] = @16) */
  for (rr=w36+11, ss=(&w16); rr!=w36+12; rr+=1) *rr = *ss++;
  /* #166: @16 = 0.9752 */
  w16 = 9.7519999999999996e-01;
  /* #167: (@36[15] = @16) */
  for (rr=w36+15, ss=(&w16); rr!=w36+16; rr+=1) *rr = *ss++;
  /* #168: @31 = mac(@30,@36,@31) */
  for (i=0, rr=w31; i<4; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w30+j, tt=w36+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #169: @31 = (-@31) */
  for (i=0, rr=w31, cs=w31; i<16; ++i) *rr++ = (- *cs++ );
  /* #170: @32 = horzcat(@34, @31) */
  rr=w32;
  for (i=0, cs=w34; i<28; ++i) *rr++ = *cs++;
  for (i=0, cs=w31; i<16; ++i) *rr++ = *cs++;
  /* #171: @37 = @32' */
  for (i=0, rr=w37, cs=w32; i<11; ++i) for (j=0; j<4; ++j) rr[i+j*11] = *cs++;
  /* #172: @38 = horzcat(@20, @33, @37) */
  rr=w38;
  for (i=0, cs=w20; i<33; ++i) *rr++ = *cs++;
  for (i=0, cs=w33; i<44; ++i) *rr++ = *cs++;
  for (i=0, cs=w37; i<44; ++i) *rr++ = *cs++;
  /* #173: @39 = @38' */
  for (i=0, rr=w39, cs=w38; i<11; ++i) for (j=0; j<11; ++j) rr[i+j*11] = *cs++;
  /* #174: @16 = input[0][2] */
  w16 = arg[0] ? arg[0][2] : 0;
  /* #175: @22 = input[0][7] */
  w22 = arg[0] ? arg[0][7] : 0;
  /* #176: @23 = input[0][8] */
  w23 = arg[0] ? arg[0][8] : 0;
  /* #177: @24 = input[0][9] */
  w24 = arg[0] ? arg[0][9] : 0;
  /* #178: @40 = vertcat(@2, @6, @16, @7, @8, @9, @4, @22, @23, @24, @25) */
  rr=w40;
  *rr++ = w2;
  *rr++ = w6;
  *rr++ = w16;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w4;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w24;
  *rr++ = w25;
  /* #179: @10 = mac(@39,@40,@10) */
  for (i=0, rr=w10; i<1; ++i) for (j=0; j<11; ++j, ++rr) for (k=0, ss=w39+j, tt=w40+i*11; k<11; ++k) *rr += ss[k*11]**tt++;
  /* #180: @0 = mac(@1,@10,@0) */
  casadi_mtimes(w1, casadi_s6, w10, casadi_s5, (&w0), casadi_s4, w, 0);
  /* #181: @2 = 0 */
  w2 = 0.;
  /* #182: @15 = zeros(1x4) */
  casadi_clear(w15, 4);
  /* #183: @34 = zeros(4x7) */
  casadi_clear(w34, 28);
  /* #184: @31 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c1, 16, w31);
  /* #185: @31 = (@35\@31) */
  rr = w31;
  ss = w35;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[10], r[10], beta[4], w[8];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 4, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #186: @35 = @31' */
  for (i=0, rr=w35, cs=w31; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #187: @33 = horzcat(@34, @35) */
  rr=w33;
  for (i=0, cs=w34; i<28; ++i) *rr++ = *cs++;
  for (i=0, cs=w35; i<16; ++i) *rr++ = *cs++;
  /* #188: @37 = @33' */
  for (i=0, rr=w37, cs=w33; i<11; ++i) for (j=0; j<4; ++j) rr[i+j*11] = *cs++;
  /* #189: @15 = mac(@1,@37,@15) */
  casadi_mtimes(w1, casadi_s6, w37, casadi_s8, w15, casadi_s7, w, 0);
  /* #190: @6 = input[1][0] */
  w6 = arg[1] ? arg[1][0] : 0;
  /* #191: @16 = input[1][1] */
  w16 = arg[1] ? arg[1][1] : 0;
  /* #192: @7 = input[1][2] */
  w7 = arg[1] ? arg[1][2] : 0;
  /* #193: @8 = input[1][3] */
  w8 = arg[1] ? arg[1][3] : 0;
  /* #194: @27 = vertcat(@6, @16, @7, @8) */
  rr=w27;
  *rr++ = w6;
  *rr++ = w16;
  *rr++ = w7;
  *rr++ = w8;
  /* #195: @2 = mac(@15,@27,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w15+j, tt=w27+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #196: @0 = (@0+@2) */
  w0 += w2;
  /* #197: @2 = 0.8 */
  w2 = 8.0000000000000004e-01;
  /* #198: @2 = (@2-@3) */
  w2 -= w3;
  /* #199: @0 = (@0+@2) */
  w0 += w2;
  /* #200: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_constr_h_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_constr_h_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_constr_h_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_constr_h_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_constr_h_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_constr_h_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    case 1: return casadi_s9;
    case 2: return casadi_s10;
    case 3: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_constr_h_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 15;
  if (sz_res) *sz_res = 12;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 679;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
