#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#include <mkl.h>
#include <mkl_types.h>
#include <mkl_blas.h>
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#include <mkl_service.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <assert.h>
#include <popt.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permute.h>
#include "Measure.hpp"
#include "StructFactor.hpp"
#include "functions.h"
#include "Utils.hpp"

const int NumberOfSpinsPerUnitCell=16;
const int SpinPerPrimitiveCell=4;

// It is specific to the pyrochlore lattice with extended unit
// cell. The list below gives the sublattices that can be updated at
// the same time.

#define IndependentSubLattices {{0, 4, 8, 12}, {0, 14, 8, 6}, {0, 14, 7, 9}, {0, 11, 13, 6}, {0, 11, 7, 12}, {0, 4, 13, 9}, \
      {1, 5, 13, 9}, {1, 15, 8, 6}, {1, 15, 7, 9}, {1, 10, 13, 6}, {1, 10, 7, 12}, {1, 5, 8, 12}, \
	{2, 10, 14, 6}, {2, 15, 11, 6}, {2, 15, 4, 9}, {2, 10, 4, 12}, {2, 5, 14, 9}, {2, 5, 11, 12}, \
	  {3, 15, 11, 7}, {3, 15, 4, 8}, {3, 10, 14, 7}, {3, 10, 4, 13}, {3, 5, 14, 8}, {3, 5, 11, 13}}

#define NumberOfParallelUpdates 24
#define NumberOfSubLatticesPerUpdate 4
const int RatioUniCellSubLattices = NumberOfSpinsPerUnitCell/NumberOfSubLatticesPerUpdate;


#undef POTTS

__constant__ int __CouplingType[1];
#ifndef _ANTIFERRO_
__constant__ TYPE __CouplingMatrices[9*SpinPerPrimitiveCell*SpinPerPrimitiveCell];
#endif
__constant__ int __MagneticField[1];
__constant__ TYPE __B[12];
__constant__ TYPE _g1[1];
__constant__ TYPE _g2[1];

void Mc(dev *device);
GLOBAL void MCMultipleTemperaturesPerSystem(const TYPE *Rng,
					    const TYPE *CosThetaSolid,
					    const TYPE *Temperature,
					    const unsigned int *Update,
					    const int ReduceAngle,
					    int *res,
					    TYPE *spin);

GLOBAL void OverRelaxationStep(const TYPE *Rng, const unsigned int *Update, TYPE *spin);
GLOBAL void OverRelaxationByPiStep(const unsigned int *Update, TYPE *spin);
GLOBAL void MCKernel4x4x4(const TYPE *Rng, const TYPE *CosThetaSolid,
			  const TYPE *Temperature, const unsigned int *Update,
			  const int ReduceAngle, int *res, TYPE *spin);
#include "Math.cu"


void Mc(dev *device)
{ 
  if(device[0].RngSize < RatioUniCellSubLattices*1000) {
    if(device[0].Rng != NULL)
      cudaFree(device[0].Rng);
    device[0].RngSize = RatioUniCellSubLattices*1000;
    cudaMalloc(&device[0].Rng, device[0].RngSize*sizeof(int));
  }

  for(int s=0;s<device[0].steps;s+=1000) {
    /* curandGenerate(device[0].Gen, device[0].Rng, ); */
    
    // make sure everything is alright
    
    cudaStreamSynchronize(device[0].streamMcpy[0]);
    int stp = device[0].steps;
    if(device[0].steps > 1000)
      stp = 1000;
    for(int m=0;m<stp;m++) {
      GenerateNewRandomNumbers(device, 0, 4*device[0].nspins, device[0].NumberOfTemperatures);
      switch(device[0].n.x) {
      case 4:
	MCKernel4x4x4<<<device[0].Blocks,BlockSpin>>>(device[0].RngDev,
						      device[0].CosThetaSolid,
						      device[0].Temperatures,
						      device[0].Rng+s+m*RatioUniCellSubLattices,
						      device[0].ReductionAngle,
						      device[0].DevResults,
						      device[0].Spins);
	break;
      default:
	for(int s=0;s<RatioUniCellSubLattices;s++) {
	  
	  // I update NumberOfSubLattices spins per unit cell in one go.
	  
	  const size_t offset = s*device[0].ncell*4*device[0].NumberOfTemperatures*NumberOfSubLatticesPerUpdate;
	  MCMultipleTemperaturesPerSystem<<<device[0].Blocks,BlockSpin>>>(device[0].RngDev+offset,
									  device[0].CosThetaSolid,
									  device[0].Temperatures,
									  device[0].Rng+s+m*RatioUniCellSubLattices,
									  device[0].ReductionAngle,
									  device[0].DevResults,
									  device[0].Spins);
	}
	if(device[0].OverRelaxation) {
	  for(auto over=0;(over<device[0].OverRelaxation);over++) {
	    for(auto s=0;s<RatioUniCellSubLattices;s++) {
	      // I update NumberOfSubLattices spins per unit cell in one go.
	      if(device[0].SharedMemory)
		OverRelaxationByPiStep<<<device[0].Blocks,BlockSpin,device[0].SharedMemory>>>(device[0].Rng+s+m*RatioUniCellSubLattices, device[0].Spins);
	      else
		OverRelaxationByPiStep<<<device[0].Blocks,BlockSpin>>>(device[0].Rng+s+m*RatioUniCellSubLattices, device[0].Spins);
	    }
	  }
	}
      }
    }
    break;
  }
}




#define _3D_
TYPE J1=-0.09/0.29;
TYPE J2=-0.22/0.29;
TYPE J3=-1.00; // (0.29)
TYPE J4=0.0;
TYPE gxy = 1.0;
TYPE gz = 1.0;
TYPE FieldOnSite[12];

#define links4 {{{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 0, 0, 2}, {0, 1, 0, 2}, {0, 0, 0, 3}, {1, 0, 0, 3}}, \
	      {{0, 0, -1, 0}, {0, 0, 0, 0}, {0, 0, 0, 2}, {0, 1, -1, 2}, {0, 0, 0, 3}, {1, 0, -1, 3}}, \
	      {{0, -1, 0, 0}, {0, 0, 0, 0}, {0, -1, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 3}, {1, -1, 0, 3}}, \
	      {{-1, 0, 0, 0}, {0, 0, 0, 0}, {-1, 0, 1, 1}, {0, 0, 0, 1}, {-1, 1, 0, 2}, {0, 0, 0, 2}}}

#define links16 {{{0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 0, 3}, {0, -1, -1, 10}, {-1, -1, 0, 5}, {-1, 0, -1, 15}}, \
      {{0, 0, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, 3}, {0, 0, 0, 4}, {0, 0, -1, 14}, {0, 0, -1, 11}}, \
	{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 3}, {0, 0, 0, 8}, {-1, 0, 0, 13}, {-1, 0, 0, 7}}, \
	  {{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}, {0, -1, 0, 9}, {0, -1, 0, 6}, {0, 0, 0, 12}}, \
	    {{0, 0, 0, 5}, {0, 0, 0, 6}, {0, 0, 0, 7}, {0, 0, 0, 1}, {0, 0, -1, 14}, {0, 0, -1, 11}}, \
	      {{0, 0, 0, 4}, {0, 0, 0, 6}, {0, 0, 0, 7}, {1, 1, 0, 0}, {1, 0, -1, 10}, {0, 1, -1, 15}}, \
		{{0, 0, 0, 4}, {0, 0, 0, 5}, {0, 0, 0, 7}, {0, 1, 0, 3}, {0, 1, 0, 12}, {0, 0, 0, 9}}, \
		  {{0, 0, 0, 4}, {0, 0, 0, 5}, {0, 0, 0, 6}, {1, 0, 0, 2}, {1, 0, 0, 8}, {0, 0, 0, 13}}, \
		    {{0, 0, 0, 9}, {0, 0, 0, 10}, {0, 0, 0, 11}, {-1, 0, 0, 7}, {0, 0, 0, 2}, {-1, 0, 0, 13}}, \
		      {{0, 0, 0, 8}, {0, 0, 0, 10}, {0, 0, 0, 11}, {0, 1, 0, 3}, {0, 0, 0, 6},  {0, 1, 0,  12}}, \
			{{0, 0, 0, 8}, {0, 0, 0, 9}, {0, 0, 0, 11}, {0, 1, 1, 0}, {-1, 0, 1, 5}, {-1, 1, 0, 15}}, \
			  { {0, 0, 0, 8}, {0, 0, 0, 9}, {0, 0, 0, 10}, {0, 0, 1, 1}, {0, 0, 1, 4}, {0, 0, 0, 14}}, \
			    {{0, 0, 0, 13}, {0, 0, 0, 14}, {0, 0, 0, 15}, {0, 0, 0, 3}, {0, -1, 0, 6}, {0, -1, 0, 9}}, \
			      {{0, 0, 0, 12}, {0, 0, 0, 14}, {0, 0, 0, 15}, {1, 0, 0, 2}, {1, 0, 0, 8}, {0, 0, 0, 7}}, \
				{{0, 0, 0, 12}, {0, 0, 0, 13}, {0, 0, 0, 15}, {0, 0, 0, 11}, {0, 0, 1, 1}, {0, 0, 1, 4}}, \
				  {{0, 0, 0, 12}, {0, 0, 0, 13}, {0, 0, 0, 14}, {1, 0, 1, 0}, {0, -1, 1, 5}, {1, -1, 0, 10}}}

#define couplings_const {{{0, 0, 0, 0, 0, 0, 0, 0, 0},			\
	{-J1, -J3, J4, -J3, -J1, J4, -J4, -J4, -J2},			\
	  {-J2, -J4, -J4, J4, -J1, -J3, J4, -J3, -J1},			\
	    {-J1, J4, -J3, -J4, -J2, -J4, -J3, J4, -J1}},		\
      {{-J1, -J3, -J4, -J3, -J1, -J4, J4, J4, -J2},			\
	  {0, 0, 0, 0, 0, 0, 0, 0, 0},					\
	    {-J1, J4, J3, -J4, -J2, J4, J3, -J4, -J1},			\
	      {-J2, -J4, J4, J4, -J1, J3, -J4, J3, -J1}},		\
	{{-J2, J4, J4, -J4, -J1, -J3, -J4, -J3, -J1},			\
	    {-J1, -J4, J3, J4, -J2, -J4, J3, J4, -J1},			\
	      {0, 0, 0, 0, 0, 0, 0, 0, 0},				\
		{-J1, J3, -J4, J3, -J1, J4, J4, -J4, -J2}},		\
	  {{-J1, -J4, -J3, J4, -J2, J4, -J3, -J4, -J1},                 \
	      {-J2, J4, -J4, -J4, -J1, J3, J4, J3, -J1},		\
		{-J1, J3, J4, J3, -J1, -J4, -J4, J4, -J2},		\
		  {0, 0, 0, 0, 0, 0, 0, 0, 0}}}

/* FILE *Energy; */
const int Neighbors4[4][6][4] = links4;
const int Neighbors16[16][6][4] = links16;
const int NumberOfNeighbors[4] = {6, 6, 6, 6};
const int CouplingType = 9 ;
int NumberOfLinks = 0;
TYPE *SymmetryMatrix;


__global__ void InitialState(TYPE *spin)
{
  const int x = threadIdx.x + blockDim.x*blockIdx.x;
  const int y = threadIdx.y + blockDim.y*blockIdx.y;
  const int z = threadIdx.z + blockDim.z*blockIdx.z;

  const int Cell = (z *gridDim.y*blockDim.y + y) * gridDim.x * blockDim.x + x;
  TYPE *Sx = spin;
  TYPE *Sy = spin + gridDim.x * gridDim.y *gridDim.z * blockDim.x  * blockDim.y * blockDim.z; 
  TYPE *Sz = spin + 2*gridDim.x * gridDim.y *gridDim.z * blockDim.x  * blockDim.y * blockDim.z; 
  Sx[Cell] = 1.0;
  Sy[Cell] = 0.0;
  Sz[Cell] = 0.0;
}

__device__ __inline__ void CalculateLocalExchangeField(const int4 n,
						       const int ncell,
						       const int3 thd,
						       const int spinIndex,
						       const TYPE *Sx,
						       const TYPE *Sy,
						       const TYPE *Sz,
						       TYPE3 *ExchangeField)
{
  const int site = spinIndex%4;
  if(__MagneticField[0]) {
    ExchangeField->x = __B[3*(site)];
    ExchangeField->y = __B[3*(site)+1];
    ExchangeField->z = __B[3*(site)+2];
  }

  const int *link = &__links[__NumberOfNeighbors[site]*spinIndex];

#pragma unroll
  for(auto s=0;s<6;s+=2) {
    int Id1 = CalculateNeighbor3D(link+s, thd, n, NULL);
    int ni = ((signed char *)(link+s))[3];
    // index of the neighboring spin
    TYPE3 tp1;
    tp1.x = Sx[Id1];
    tp1.y = Sy[Id1];
    tp1.z = Sz[Id1];
#ifndef _ANTIFERRO_
    const TYPE *Cp = __CouplingMatrices+9*(4*site+(ni%4));
#endif

    // check if it is inside the block or outside
    int ids = CalculateNeighbor3D(link+s+1, thd, n, NULL);
    ni = ((signed char *)(&link[s+1]))[3];
    // index of the neighboring spin
    TYPE3 tp;
    tp.x = Sx[ids];
    tp.y = Sy[ids];
    tp.z = Sz[ids];
#ifndef _ANTIFERRO_
    const TYPE *Cp1 = __CouplingMatrices+9*(4*site+(ni%4));

    // Spin one-half

    ExchangeField->x += 0.5*(Cp[0]*tp1.x + Cp[1]*tp1.y + Cp[2]*tp1.z);
    ExchangeField->y += 0.5*(Cp[3]*tp1.x + Cp[4]*tp1.y + Cp[5]*tp1.z);
    ExchangeField->z += 0.5*(Cp[6]*tp1.x + Cp[7]*tp1.y + Cp[8]*tp1.z);

    // Spin one-half

    ExchangeField->x += 0.5*(Cp1[0]*tp.x + Cp1[1]*tp.y + Cp1[2]*tp.z);
    ExchangeField->y += 0.5*(Cp1[3]*tp.x + Cp1[4]*tp.y + Cp1[5]*tp.z);
    ExchangeField->z += 0.5*(Cp1[6]*tp.x + Cp1[7]*tp.y + Cp1[8]*tp.z);
#else
    ExchangeField->x +=tp1.x;
    ExchangeField->y +=tp1.x;
    ExchangeField->z +=tp1.x;
    
    // Spin one-half

    ExchangeField->x += tp.x;
    ExchangeField->y += tp.y;
    ExchangeField->z += tp.z;
#endif
  }
}

GLOBAL void MagnetizationGPU(const TYPE *spin, TYPE *Mx, TYPE *My, TYPE *Mz)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, cell, ncell, Offset;
  int3 thd;
  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &cell, &ncell, &Offset);
  const TYPE g1[3][3] = {{_g1[0], _g2[0], _g2[0]}, {_g2[0], _g1[0], _g2[0]}, {_g2[0], _g2[0], _g1[0]}};
  const TYPE g2[3][3] = {{_g1[0], _g2[0], -_g2[0]}, {_g2[0], _g1[0], -_g2[0]}, {-_g2[0], -_g2[0], _g1[0]}};
  const TYPE g3[3][3] = {{_g1[0], -_g2[0], -_g2[0]}, {-_g2[0], _g1[0], _g2[0]}, {-_g2[0], _g2[0], _g1[0]}};
  const TYPE g4[3][3] = {{_g1[0], -_g2[0], _g2[0]}, {-_g2[0], _g1[0], -_g2[0]}, {_g2[0], -_g2[0], _g1[0]}};

  const TYPE *Sx = spin + 3*Offset;
  const TYPE *Sy = Sx + ncell*n.w;
  const TYPE *Sz = Sy + ncell*n.w;
  
  TYPE3 mm4;
  TYPE3 s0, s1, s2, s3;
  s0.x=0;
  s0.y=0;
  s0.z=0;
  s1.x=0;
  s1.y=0;
  s1.z=0;
  s2.x=0;
  s2.y=0;
  s2.z=0;
  s3.x=0;
  s3.y=0;
  s3.z=0;

  for(auto s=0;s<4;s++) {
    s0.x += Sx[cell+4*s*ncell];
    s0.y += Sy[cell+4*s*ncell];
    s0.z += Sz[cell+4*s*ncell];
    s1.x += Sx[cell+(4*s+1)*ncell];
    s1.y += Sy[cell+(4*s+1)*ncell];
    s1.z += Sz[cell+(4*s+1)*ncell];
    s2.x += Sx[cell+(4*s+2)*ncell];
    s2.y += Sy[cell+(4*s+2)*ncell];
    s2.z += Sz[cell+(4*s+2)*ncell];
    s3.x += Sx[cell+(4*s+3)*ncell];
    s3.y += Sy[cell+(4*s+3)*ncell];
    s3.z += Sz[cell+(4*s+3)*ncell];
  }


  mm4.x = g1[0][0] * s0.x + g1[0][1] * s0.y + g1[0][2] * s0.z;
  mm4.y = g1[1][0] * s0.x + g1[1][1] * s0.y + g1[1][2] * s0.z;
  mm4.z = g1[2][0] * s0.x + g1[2][1] * s0.y + g1[2][2] * s0.z;

  mm4.x += g2[0][0] * s1.x + g2[0][1] * s1.y + g2[0][2] * s1.z;
  mm4.y += g2[1][0] * s1.x + g2[1][1] * s1.y + g2[1][2] * s1.z;
  mm4.z += g2[2][0] * s1.x + g2[2][1] * s1.y + g2[2][2] * s1.z;

  mm4.x += g3[0][0] * s2.x + g3[0][1] * s2.y + g3[0][2] * s2.z;
  mm4.y += g3[1][0] * s2.x + g3[1][1] * s2.y + g3[1][2] * s2.z;
  mm4.z += g3[2][0] * s2.x + g3[2][1] * s2.y + g3[2][2] * s2.z;

  mm4.x += g4[0][0] * s3.x + g4[0][1] * s3.y + g4[0][2] * s3.z;
  mm4.y += g4[1][0] * s3.x + g4[1][1] * s3.y + g4[1][2] * s3.z;
  mm4.z += g4[2][0] * s3.x + g4[2][1] * s3.y + g4[2][2] * s3.z;

  Mx[cell + TemperatureNumber*ncell] = mm4.x;
  My[cell + TemperatureNumber*ncell] = mm4.y;
  Mz[cell + TemperatureNumber*ncell] = mm4.z;
}

GLOBAL void TotalEnergyGPU(const TYPE *spin, TYPE *Energy)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int3 thd;
  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);

  const TYPE *Sx = spin + 3*Offset;
  const TYPE *Sy = Sx + n.w*ncell;
  const TYPE *Sz = Sy + n.w*ncell;
  
  TYPE EnergyTmp = 0.0;

  /* sum over the neighbors */

  for(auto site=0;site<16;++site) {
    TYPE3 ExchangeField ={0,0,0};
    TYPE3 tmp;
    tmp.x = Sx[Cell+site*ncell];
    tmp.y = Sy[Cell+site*ncell];
    tmp.z = Sz[Cell+site*ncell];

    CalculateLocalExchangeField(n, ncell, thd, site, Sx, Sy, Sz, &ExchangeField);

    // I need to Add the zeeman term a second time because otherwise the zeeman term appears only 1/2 time in the total energy. There is a problem of double counting the energy per link

    // H = 1/2 s_i s_j + nu_b B.s_i

    // NOT

    // H= 1/2 (S_j + mu_b B)
    if(__MagneticField[0]) {
      ExchangeField.x += __B[3*(site%4)];
      ExchangeField.y += __B[3*(site%4)+1];
      ExchangeField.z += __B[3*(site%4)+2];
    }

    EnergyTmp += (tmp.x * ExchangeField.x +
		  tmp.y * ExchangeField.y +
		  tmp.z * ExchangeField.z);
  }

  // spin one-half

  Energy[Cell+TemperatureNumber*ncell] = -0.5*EnergyTmp;
}

GLOBAL void UnscrubleData(const TYPE3 *src, const dim3 n, const int nspc, TYPE3 *dst)
{
  int LocalCell = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
  int blk = (blockIdx.z*gridDim.y + blockIdx.y)*gridDim.x+blockIdx.x;
  int tht = blockDim.x*blockDim.y*blockDim.z;

  // compute the global coordinates of the cell
  int x = threadIdx.x + blockDim.x*blockIdx.x;
  int y = threadIdx.y + blockDim.y*blockIdx.y;
  int z = threadIdx.z + blockDim.z*blockIdx.z;

  // compute the global index of the cell
  int Cell = (z*n.y + y)*n.x+x;


#pragma unroll 4
  for(auto site=0;site<nspc;site++) {
      // Index of the spin
    int GlobalIndexSpin = Cell+site*n.x*n.y*n.z;

    dst[GlobalIndexSpin] = src[LocalCell + (site + blk * nspc) * tht];
  }
}
void InitializeMagneticField(const double * B, double * BSite)
{
  TYPE g1 = (2.0*gxy + gz)/3.0;
  TYPE g2 = (gz-gxy)/3.0;

  TYPE g11[3][3] = {{g1, g2, g2}, {g2, g1, g2}, {g2, g2, g1}};
  TYPE g21[3][3] = {{g1, g2, -g2}, {g2, g1, -g2}, {-g2, -g2, g1}};
  TYPE g3[3][3] = {{g1, -g2, -g2}, {-g2, g1, g2}, {-g2, g2, g1}};
  TYPE g4[3][3] = {{g1, -g2, g2}, {-g2, g1, -g2}, {g2, -g2, g1}};
  memset(BSite, 0, sizeof(TYPE)*12);
  for(auto s=0;s<3;++s) {
    for(auto s1=0;s1<3;++s1) {
      BSite[s] += g11[s][s1]*B[s1];
    }
  }
  for(auto s=0;s<3;++s) {
    for(auto s1=0;s1<3;++s1) {
      BSite[s+3] += g21[s][s1]*B[s1];
    }
  }
  for(auto s=0;s<3;++s) {
    for(auto s1=0;s1<3;++s1) {
      BSite[s+6] += g3[s][s1]*B[s1];
    }
  }
  for(auto s=0;s<3;++s) {
    for(auto s1=0;s1<3;++s1) {
      BSite[s+9] += g4[s][s1]*B[s1];
    }
  }

  for(auto s=0;s<4;++s)
    printf("%.5lf %.5lf %.5lf\n", BSite[3*s], BSite[3*s+1], BSite[3*s+2]);
  // B is in Tesla so I convert it to meV
  for(auto s=0;s<12;s++)
    BSite[s] *= 0.0578838261305;
}


void Magnetization(const int NumberOfTemperatures, std::vector<dev> &device, TYPE *Energy)
{
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    int SizeCellDev = device[d].NumberOfTemperatures*device[d].ncell;
    device[d].CurrentStream = device[d].streamMc[3];
    MagnetizationGPU<<<device[d].Blocks,BlockSpin,device[d].SharedMemory, device[d].CurrentStream>>>(device[d].DevSpins,
												     device[d].Red,
												     device[d].Red+SizeCellDev,
												     device[d].Red+2*SizeCellDev);
  }

  Reduce(3, NumberOfTemperatures, device, Energy);
}

double TotalEnergy(const TYPE *spin, const dim3 n)
{
  double tmp = 0;
  const TYPE Couplings[4][4][9] = couplings_const;
  const TYPE *Sx = spin;
  const TYPE *Sy = Sx + NumberOfSpinsPerUnitCell*n.x*n.y*n.z;
  const TYPE *Sz = Sy + NumberOfSpinsPerUnitCell*n.x*n.y*n.z;
  int x,y,z,site ;
  for(z=0;z<n.z;z++) {
    for(y=0;y<n.y;y++) {
      for(x=0;x<n.x;x++) {
	for(site=0;site<NumberOfSpinsPerUnitCell;site++) {
	  TYPE ex =  2.0*FieldOnSite[3*(site%4)];
	  TYPE ey =  2.0*FieldOnSite[3*(site%4)+1];
	  TYPE ez =  2.0*FieldOnSite[3*(site%4)+2];
	  int Id = ((y + z*n.y)*n.x+x) + site*n.x*n.y*n.z;
	  for(auto se=0;se<6;se++) {
	    int x1 = x + Neighbors16[site][se][0] + n.x;
            while(x1>= n.x)
              x1-= n.x;
            int y1 = y + Neighbors16[site][se][1] + n.y;
            while(y1>= n.y)
              y1-= n.y;
	    int z1 = z + Neighbors16[site][se][2] + n.z;
            while(z1>= n.z)
              z1-= n.z;
            
	    int ne = Neighbors16[site][se][3];
	    int Idd = ((z1*n.y+y1)*n.x+x1) + ne*n.x*n.y*n.z;
	    int si = site%SpinPerPrimitiveCell;
	    int sj = ne%SpinPerPrimitiveCell;
#ifndef _ANTIFERRO_
	    const TYPE *cp = Couplings[si][sj];
	    ex += 0.5*(cp[0]*Sx[Idd] +
		       cp[1]*Sy[Idd] +
		       cp[2]*Sz[Idd]);
	    ey += 0.5*(cp[3]*Sx[Idd] +
		       cp[4]*Sy[Idd] +
		       cp[5]*Sz[Idd]);
	    ez += 0.5*(cp[6]*Sx[Idd] +
		       cp[7]*Sy[Idd] +
		       cp[8]*Sz[Idd]);
#else
	    ex += 0.5 * cp[0];
	    ey += 0.5 * cp[1];
	    ez += 0.5 * cp[2];
#endif
	  }
	  tmp -= (ex * Sx[Id] + ey * Sy[Id] + ez * Sz[Id]);
	}
      }
    }
  }

  return tmp*S2;
}


GLOBAL void Measurement(const TYPE *spin, TYPE *OrderParameters)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, cell, ncell, Offset;
  int3 thd;
  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &cell, &ncell, &Offset);

  const TYPE *Sx = spin + 3*Offset;
  const TYPE *Sy = Sx + 16*ncell;
  const TYPE *Sz = Sy + 16*ncell;
  TYPE mm1 = 0;
  TYPE2 mm2;
  TYPE3 mm3;
  TYPE3 mm4;
  TYPE3 mm5;
  mm2.x = 0;
  mm2.y = 0;
  mm3.x = 0;
  mm3.y = 0;
  mm3.z = 0;
  mm4.x = 0;
  mm4.y = 0;
  mm4.z = 0;
  mm5.x = 0;
  mm5.y = 0;
  mm5.z = 0;
  TYPE3 s0, s1, s2, s3;
  s0.x=0;
  s0.y=0;
  s0.z=0;
  s1.x=0;
  s1.y=0;
  s1.z=0;
  s2.x=0;
  s2.y=0;
  s2.z=0;
  s3.x=0;
  s3.y=0;
  s3.z=0;
  for(auto s=0;s<4;s++) {
    s0.x += Sx[cell+4*s*ncell];
    s0.y += Sy[cell+4*s*ncell];
    s0.z += Sz[cell+4*s*ncell];
    s1.x += Sx[cell+(4*s+1)*ncell];
    s1.y += Sy[cell+(4*s+1)*ncell];
    s1.z += Sz[cell+(4*s+1)*ncell];
    s2.x += Sx[cell+(4*s+2)*ncell];
    s2.y += Sy[cell+(4*s+2)*ncell];
    s2.z += Sz[cell+(4*s+2)*ncell];
    s3.x += Sx[cell+(4*s+3)*ncell];
    s3.y += Sy[cell+(4*s+3)*ncell];
    s3.z += Sz[cell+(4*s+3)*ncell];
  }

  mm1= 0.5773502691896258*s0.x + 0.5773502691896258*s0.y + 0.5773502691896258*s0.z -
    0.5773502691896258*s1.x - 0.5773502691896258*s1.y + 0.5773502691896258*s1.z +
    0.5773502691896258*s2.x - 0.5773502691896258*s2.y - 0.5773502691896258*s2.z -
    0.5773502691896258*s3.x + 0.5773502691896258*s3.y - 0.5773502691896258*s3.z;

  mm2.x= -0.816496580927726*s0.x + 0.4082482904638631*s0.y + 0.4082482904638631*s0.z +
    0.816496580927726*s1.x - 0.4082482904638631*s1.y + 0.4082482904638631*s1.z -
    0.816496580927726*s2.x - 0.4082482904638631*s2.y - 0.4082482904638631*s2.z +
    0.816496580927726*s3.x + 0.4082482904638631*s3.y - 0.4082482904638631*s3.z;
  mm2.y=-0.7071067811865475*s0.y + 0.7071067811865475*s0.z + 0.7071067811865475*s1.y +
    0.7071067811865475*s1.z + 0.7071067811865475*s2.y - 0.7071067811865475*s2.z -
    0.7071067811865475*s3.y - 0.7071067811865475*s3.z;

  mm3.x= -0.7071067811865475*s0.y + 0.7071067811865475*s0.z -
    0.7071067811865475*s1.y - 0.7071067811865475*s1.z +
    0.7071067811865475*s2.y - 0.7071067811865475*s2.z +
    0.7071067811865475*s3.y + 0.7071067811865475*s3.z;

  mm3.y= 0.7071067811865475*s0.x - 0.7071067811865475*s0.z +
    0.7071067811865475*s1.x + 0.7071067811865475*s1.z -
    0.7071067811865475*s2.x - 0.7071067811865475*s2.z -
    0.7071067811865475*s3.x + 0.7071067811865475*s3.z;

  mm3.z= -0.7071067811865475*s0.x + 0.7071067811865475*s0.y + 0.7071067811865475*s1.x -
    0.7071067811865475*s1.y + 0.7071067811865475*s2.x + 0.7071067811865475*s2.y -
    0.7071067811865475*s3.x - 0.7071067811865475*s3.y;

  mm4.x = s0.x + s1.x + s2.x + s3.x;
  mm4.y = s0.y + s1.y + s2.y + s3.y;
  mm4.z = s0.z + s1.z + s2.z + s3.z;

  mm5.x = -0.7071067811865475*s0.y - 0.7071067811865475*s0.z -
    0.7071067811865475*s1.y + 0.7071067811865475*s1.z +
    0.7071067811865475*s2.y + 0.7071067811865475*s2.z +
    0.7071067811865475*s3.y - 0.7071067811865475*s3.z;

  mm5.y = -0.7071067811865475*s0.x - 0.7071067811865475*s0.z -
    0.7071067811865475*s1.x + 0.7071067811865475*s1.z +
    0.7071067811865475*s2.x - 0.7071067811865475*s2.z +
    0.7071067811865475*s3.x + 0.7071067811865475*s3.z;

  mm5.z = -0.7071067811865475*s0.x - 0.7071067811865475*s0.y +
    0.7071067811865475*s1.x + 0.7071067811865475*s1.y +
    0.7071067811865475*s2.x - 0.7071067811865475*s2.y -
    0.7071067811865475*s3.x + 0.7071067811865475*s3.y;

  TYPE *Order = OrderParameters + 12 * TemperatureNumber * ncell;
  //Ord[T][m][cell]

  Order[cell ] = mm1;
  Order[cell + ncell] = mm2.x;
  Order[cell + 2*ncell] = mm2.y;
  Order[cell + 3*ncell] = mm3.x;
  Order[cell + 4*ncell] = mm3.y;
  Order[cell + 5*ncell] = mm3.z;
  Order[cell + 6*ncell] = mm4.x;
  Order[cell + 7*ncell] = mm4.y;
  Order[cell + 8*ncell] = mm4.z;
  Order[cell + 9*ncell] = mm5.x;
  Order[cell + 10*ncell] = mm5.y;
  Order[cell + 11*ncell] = mm5.z;
}

void InitializeCouplings3D(const int *nt, const int nspc, const int SpinPerPrimitiveCells, const int *neighbors, const int MagneticField, const double *B, std::vector<dev> &device)
{
  TYPE Couplings[4][4][9] = couplings_const;
  char Indep[NumberOfParallelUpdates][4] = IndependentSubLattices;
  int NumberOfLinks = neighbors[0];
  int PrimitiveCell = nspc/SpinPerPrimitiveCells;
  for(auto s=1;s<SpinPerPrimitiveCell;s++)
    NumberOfLinks += neighbors[s];

  int *tmp = (int *)malloc(sizeof(int) * PrimitiveCell * NumberOfLinks);
  int *tmp1 = (int *)malloc(sizeof(int) * SpinPerPrimitiveCells * 6);

  int ni = 0;
  for(auto s=0;s<nspc;s++) {
    signed char *tc = (signed char *)tmp;
    for(auto s1=0;s1<neighbors[s%SpinPerPrimitiveCell];s1++,ni++) {
      if(PrimitiveCell == 1) {
	tc[4*ni] = Neighbors4[s][s1][0];
	tc[4*ni+1] = Neighbors4[s][s1][1];
	tc[4*ni+2] = Neighbors4[s][s1][2];
	tc[4*ni+3] = Neighbors4[s][s1][3];
      } else {
	tc[4*ni] = Neighbors16[s][s1][0];
	tc[4*ni+1] = Neighbors16[s][s1][1];
	tc[4*ni+2] = Neighbors16[s][s1][2];
	tc[4*ni+3] = Neighbors16[s][s1][3];
      }
    }
  }

  TYPE *tmpd = (TYPE *)malloc(sizeof(TYPE)* 16 * 9);

#ifndef _ANTIFERRO_
  for(auto s=0;s<SpinPerPrimitiveCells;s++)
    for(auto s1=0;s1<SpinPerPrimitiveCells;s1++)
      memcpy(tmpd+9*(SpinPerPrimitiveCells*s+s1), Couplings[s][s1], sizeof(TYPE)*9);
#endif

  memset( FieldOnSite, 0 , sizeof(double)*12);
  for(auto d=0;d<device.size();d++) {
    device[d].nspc = nspc;

    cudaSetDevice(d);
    cudaMemcpyToSymbol(__SpinPerPrimitiveCell, &SpinPerPrimitiveCells, sizeof(int));
    cudaMemcpyToSymbol(__PrimitiveCells, &PrimitiveCell, sizeof(int));
    cudaMemcpyToSymbol(__links, tmp, sizeof(int)*nspc*6);
    cudaMemcpyToSymbol(__NumberOfSpinsPerUnitCell, &nspc, sizeof(int));
    cudaMemcpyToSymbol(__CouplingType, &CouplingType, sizeof(int));
    cudaMemcpyToSymbol(__NumberOfLinks, &NumberOfLinks, sizeof(int));
#ifndef _ANTIFERRO_
    cudaMemcpyToSymbol(__CouplingMatrices, tmpd, sizeof(TYPE)*SpinPerPrimitiveCells*SpinPerPrimitiveCells*9);
#endif
    cudaMemcpyToSymbol(__SystemSize, nt, sizeof(int)*3);
    cudaMemcpyToSymbol(__NumberOfNeighbors, NumberOfNeighbors, sizeof(int)*SpinPerPrimitiveCells);
    TYPE g1 = (2.0*gxy + gz)/3.0;
    TYPE g2 = (gz-gxy)/3.0;

    cudaMemcpyToSymbol(_g1, &g1, sizeof(TYPE));
    cudaMemcpyToSymbol(_g2, &g2, sizeof(TYPE));
    if(MagneticField) {
      InitializeMagneticField(B, FieldOnSite);
      cudaMemcpyToSymbol(__B, FieldOnSite, sizeof(TYPE)*3*4);
    }
    cudaMemcpyToSymbol(__MagneticField, &MagneticField, sizeof(int));
    cudaMemcpyToSymbol(__IndependentSites, Indep[0], sizeof(int)*NumberOfParallelUpdates);
  }
  free(tmp);
  free(tmpd);
}

GLOBAL void MCKernel4x4x4(const TYPE *Rng,
                          const TYPE *CosThetaSolid,
                          const TYPE *Temperature,
                          const unsigned int *Update,
                          const int ReduceAngle,
                          int *res,
                          TYPE *spin)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int3 thd;

  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  const TYPE InverseTemperature =   0.5/Temperature[TemperatureNumber];
  int mask = 0;

  TYPE *Sx = spin + 3*Offset;
  TYPE *Sy = Sx + 16*ncell;
  TYPE *Sz = Sy + 16*ncell;

  // shared memory is statically allocated
  __shared__ TYPE sx[1024];
  __shared__ TYPE sy[1024];
  __shared__ TYPE sz[1024];
  
  for(int site=0;site<16;site++) {
    int *link = &__links[6*site];
    const TYPE *rg = Rng + TemperatureNumber*ncell*16*4;

    TYPE4 rng;
    rng.x = rg[(LocalCell+site)*4];
    rng.y = rg[(LocalCell+site)*4+1];
    rng.z = rg[(LocalCell+site)*4+2];
    rng.w = rg[(LocalCell+site)*4+3];
    
    TYPE3 ExchangeField;

#ifdef USE_DOUBLE
    ExchangeField.x = 0.0;
    ExchangeField.y = 0.0;
    ExchangeField.z = 0.0;
#else
    ExchangeField.x = 0.0f;
    ExchangeField.y = 0.0f;
    ExchangeField.z = 0.0f;
#endif
    TYPE3 out;
    TYPE3 tmp;
    if(mask&(1<<site)) {
      // The information is already in shared memory so retrieve it.
      tmp.x = sx[LocalCell + site*ncell]; 
      tmp.y = sy[LocalCell + site*ncell]; 
      tmp.z = sz[LocalCell + site*ncell]; 
    } else {
      // Ok we need to access global memory
      tmp.x = Sx[Cell + site*ncell]; 
      tmp.y = Sy[Cell + site*ncell]; 
      tmp.z = Sz[Cell + site*ncell]; 
    }
    __Generate3DVector(rng.x, rng.y, (TYPE *)&out);
    
    for(auto nei=0;nei<6;nei++) {
      
      const int ids = CalculateNeighbor3D(link+nei, thd, n, NULL);
      int ni = ((signed char *)(link+nei))[3];
      if(!(mask&(1<<ni))) // means I do not have all information yet
        {
	  // retrieve the all sublattice
          sx[LocalCell + ni*ncell] = Sx[LocalCell + ni*ncell];
          sy[LocalCell + ni*ncell] = Sy[LocalCell + ni*ncell];
          sz[LocalCell + ni*ncell] = Sz[LocalCell + ni*ncell];
          
          __syncthreads();
	  // sublattice retrieved. No need to reread the data
          mask = mask ^ (1<<ni);
        }
      
      // we have the datas.
      TYPE3 tp;
      tp.x = sx[ids];
      tp.y = sy[ids];
      tp.z = sz[ids];

      const TYPE *Cp = __CouplingMatrices+9*(4*(site%4)+(ni%4));
      
      // Spin one-half
      
      ExchangeField.x += 0.5*(Cp[0]*tp.x + Cp[1]*tp.y + Cp[2]*tp.z);
      ExchangeField.y += 0.5*(Cp[3]*tp.x + Cp[4]*tp.y + Cp[5]*tp.z);
      ExchangeField.z += 0.5*(Cp[6]*tp.x + Cp[7]*tp.y + Cp[8]*tp.z);
    }
    
    int test = Metropolis(ExchangeField, InverseTemperature, &tmp, rng.z, &out);
    _CudaRotate3DVector(2*M_PI*rng.w,
			ExchangeField, 
			&out); 
    
    sx[LocalCell+site*ncell] = out.x;
    sy[LocalCell+site*ncell] = out.y;
    sz[LocalCell+site*ncell] = out.z;

    __syncthreads();
    if(!(mask & (1<<site))) // if shared memory is not initialized yet
      mask = mask ^ (1<<site);
  }
  
  for(int site=0;site<16;site++)
    {
      Sx[Cell+site*ncell] = sx[Cell + site*ncell];
      Sy[Cell+site*ncell] = sy[Cell + site*ncell];
      Sz[Cell+site*ncell] = sz[Cell + site*ncell];
    }

   __syncthreads();
}

GLOBAL void MCMultipleTemperaturesPerSystem(const TYPE *Rng,
					    const TYPE *CosThetaSolid,
					    const TYPE *Temperature,
					    const unsigned int *Update,
					    const int ReduceAngle,
					    int *res,
					    TYPE *spin)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int3 thd;

  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  const TYPE InverseTemperature =   1./Temperature[TemperatureNumber];
  TYPE *Sx = spin + 3*Offset;
  TYPE *Sy = Sx + 16*ncell;
  TYPE *Sz = Sy + 16*ncell;
  
  /* sum over the neighbors */
  signed char *st= (signed char *)&__IndependentSites[Update[0]%NumberOfParallelUpdates];
  TYPE3 sp[4];

#pragma unroll 4
  for(auto SubLattice=0;SubLattice<NumberOfSubLatticesPerUpdate;SubLattice++) {
    TYPE3 ExchangeField ={0,0,0};
    TYPE3 out;
    int spinIndex = st[SubLattice];
    TYPE3 tmp;
    tmp.x = Sx[Cell+spinIndex*ncell];
    tmp.y = Sy[Cell+spinIndex*ncell];
    tmp.z = Sz[Cell+spinIndex*ncell];
    const TYPE *rg = Rng + ((SubLattice + NumberOfSubLatticesPerUpdate*blk)*tht + TemperatureNumber*ncell)*4;
    TYPE4 rng;
    rng.x = rg[LocalCell];
    rng.y = rg[LocalCell+tht];
    rng.z = rg[LocalCell+2*tht];
    rng.w = rg[LocalCell+3*tht];
    __Generate3DVector(rng.x, rng.y, (TYPE *)&out);

    CalculateLocalExchangeField(n, ncell, thd, spinIndex, Sx, Sy, Sz, &ExchangeField);
    int test = Metropolis(ExchangeField, 0.5*InverseTemperature, &tmp, rng.z, &out);
    
    
    _CudaRotate3DVector(2*M_PI*rng.w,
                        ExchangeField, 
                        &out); 
    sp[SubLattice] = out;
  }

  for(auto sub=0;sub<4;sub++) {
    const int spinIndex = st[sub];
    Sx[Cell + spinIndex*ncell] = sp[sub].x;
    Sy[Cell + spinIndex*ncell] = sp[sub].y;
    Sz[Cell + spinIndex*ncell] = sp[sub].z;
  }

}

GLOBAL void MCMultipleTemperaturesPerSystemTest(const TYPE *Rng,
						const TYPE *CosThetaSolid,
						const TYPE *Temperature,
						const unsigned int *Update,
						const int ReduceAngle,
						int *res,
						TYPE *spin)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int3 thd;

  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  TYPE *Sx = spin + 3*Offset;
  TYPE *Sy = Sx + ncell*16;
  TYPE *Sz = Sy + 16*ncell;
  
  const TYPE InverseTemperature =   1./Temperature[TemperatureNumber];
  TYPE3 *SharedMem = (TYPE3 *)shared_mem;
  /* sum over the neighbors */
  signed char *st= (signed char *)&__IndependentSites[Update[0]%NumberOfParallelUpdates];
  //#pragma unroll

  for(auto SubLattice=0;SubLattice<NumberOfSubLatticesPerUpdate;SubLattice++) {
    TYPE3 ExchangeField ={0,0,0};
    TYPE3 out;
    int spinIndex = st[SubLattice];
    TYPE3 tmp;
    tmp.x = Sx[Cell+spinIndex*ncell];
    tmp.y = Sy[Cell+spinIndex*ncell];
    tmp.z = Sz[Cell+spinIndex*ncell];
    const TYPE *rg = Rng + ((SubLattice + NumberOfSubLatticesPerUpdate*blk)*tht + TemperatureNumber*ncell)*4;
    TYPE4 rng;
    int ne[6] = {0,0,0,0,0,0};
    rng.x = rg[LocalCell];
    rng.y = rg[LocalCell+tht];
    rng.z = rg[LocalCell+2*tht];
    rng.w = rg[LocalCell+3*tht];

    const int site = spinIndex%4;
    if(__MagneticField[0]) {
      ExchangeField.x = __B[3*(site)];
      ExchangeField.y = __B[3*(site)+1];
      ExchangeField.z = __B[3*(site)+2];
    }
    int *link = &__links[__NumberOfNeighbors[site]*spinIndex];

    for(auto nei=0;nei<6;nei++) {
      
      int ni = ((signed char *)(&link[nei]))[3];
      int ids = CalculateNeighbor3D(link+nei, thd, n, NULL);
      
      SharedMem[nei*tht+LocalCell].x = Sx[ids + ni*ncell];
      SharedMem[nei*tht+LocalCell].y = Sy[ids + ni*ncell];
      SharedMem[nei*tht+LocalCell].z = Sz[ids + ni*ncell];
      ne[nei] = ni;
    }
    __Generate3DVector(rng.x, rng.y, (TYPE *)&out);
    __syncthreads();

    for(auto nei=0;nei<6;nei++) {
      const TYPE3 tp1 = SharedMem[nei*tht+LocalCell];
#ifndef _ANTIFERRO_
      const TYPE *Cp = __CouplingMatrices+9*(4*site+(ne[nei] % 4));
      ExchangeField.x += 0.5*(Cp[0]*tp1.x + Cp[1]*tp1.y + Cp[2]*tp1.z);
      ExchangeField.y += 0.5*(Cp[3]*tp1.x + Cp[4]*tp1.y + Cp[5]*tp1.z);
      ExchangeField.z += 0.5*(Cp[6]*tp1.x + Cp[7]*tp1.y + Cp[8]*tp1.z);
#else
      ExchangeField.x += 0.5*tp1.x;
      ExchangeField.y += 0.5*tp1.y;
      ExchangeField.z += 0.5*tp1.z;
#endif
    }

    int test =  Metropolis(ExchangeField, 0.5*InverseTemperature, &tmp, rng.z, &out);
    if(res)
      res[Cell+ncell*TemperatureNumber]+=test;

    TYPE3 *tmp2 = ((test)?(&out):(&tmp));
    _CudaRotate3DVector(2*M_PI*rng.w,
       			ExchangeField,
       			&out);
    Sx[Cell + spinIndex*ncell] = out.x;
    Sy[Cell + spinIndex*ncell] = out.y;
    Sz[Cell + spinIndex*ncell] = out.z;
  }
}

GLOBAL void OverRelaxationByPiStep(const unsigned int *Update, TYPE *spin)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int3 thd;

  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  TYPE *Sx = spin + 3*Offset;
  TYPE *Sy = Sx + 16*ncell;
  TYPE *Sz = Sy + 16*ncell;

  /* sum over the neighbors */
  signed char *st= (signed char *)&__IndependentSites[Update[0]%NumberOfParallelUpdates];

  for(auto SubLattice=0;SubLattice<NumberOfSubLatticesPerUpdate;SubLattice++) {
    TYPE3 ExchangeField ={0,0,0};

    int spinIndex = st[SubLattice];
    TYPE3 tmp;
    tmp.x = Sx[Cell+spinIndex*ncell];
    tmp.y = Sy[Cell+spinIndex*ncell];
    tmp.z = Sz[Cell+spinIndex*ncell];


    CalculateLocalExchangeField(n, ncell, thd, spinIndex, Sx, Sy, Sz, &ExchangeField);


    _CudaRotate3DVectorByPi(ExchangeField, &tmp);
    Sx[Cell + spinIndex*ncell] = tmp.x;
    Sy[Cell + spinIndex*ncell] = tmp.y;
    Sz[Cell + spinIndex*ncell] = tmp.z;
  }
}

GLOBAL void OverRelaxationStep(const TYPE *Rng, const unsigned int *Update, TYPE *spin)
{
  int4 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int3 thd;

  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  
  TYPE *Sx = spin + 3*Offset;
  TYPE *Sy = Sx + 16*ncell;
  TYPE *Sz = Sy + 16*ncell;

  /* sum over the neighbors */
  signed char *st= (signed char *)&__IndependentSites[Update[0]%NumberOfParallelUpdates];

  for(auto SubLattice=0;SubLattice<NumberOfSubLatticesPerUpdate;SubLattice++) {
    TYPE3 ExchangeField ={0,0,0};

    int spinIndex = st[SubLattice];
    const TYPE *rg = Rng + ((SubLattice + NumberOfSubLatticesPerUpdate*blk)*tht + TemperatureNumber*ncell);
    TYPE rng = rg[LocalCell];
    TYPE3 tmp;
    tmp.x = Sx[Cell+spinIndex*ncell];
    tmp.y = Sy[Cell+spinIndex*ncell];
    tmp.z = Sz[Cell+spinIndex*ncell];

    CalculateLocalExchangeField(n, ncell, thd, spinIndex, Sx, Sy, Sz, &ExchangeField);

    _CudaRotate3DVector(2.0*M_PI*rng,
      			ExchangeField,
      			&tmp);

    Sx[Cell + spinIndex*ncell] = tmp.x;
    Sy[Cell + spinIndex*ncell] = tmp.y;
    Sz[Cell + spinIndex*ncell] = tmp.z;
  }
}

void CalculateAbsSz(const int UnitCell, 
		    const int NumberOfTemperatures, 
		    std::vector<dev> &device, 
		    TYPE *OrderParameters)
{
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    // on device
    device[d].CurrentStream = device[d].streamMc[3];
    // OrderParameters[T][m][cell]
    CudaCalculateAbsSz<<<device[d].Blocks,BlockSpin,0,device[d].CurrentStream>>>(device[d].DevSpins,
										 device[d].Red);
  }
  
  Reduce(1, NumberOfTemperatures, device, OrderParameters);
}

void CalculateAbsSumSz(const int UnitCell, 
		       const int NumberOfTemperatures, 
		       std::vector<dev> &device, 
		       TYPE *OrderParameters)
{
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    // on device
    device[d].CurrentStream = device[d].streamMc[3];
    // OrderParameters[T][m][cell]
    CudaCalculateMonopoleSquare<<<device[d].Blocks,BlockSpin,0,device[d].CurrentStream>>>(device[d].DevSpins,
											  device[d].Red);
  }
  
  Reduce(1, NumberOfTemperatures, device, OrderParameters);
}

void DoMeasurement(const int UnitCell, const int NumberOfTemperatures, std::vector<dev> &device, TYPE *OrderParameters)
{
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    // on device
    device[d].CurrentStream = device[d].streamMc[3];
    // OrderParameters[T][m][cell]
    Measurement<<<device[d].Blocks,BlockSpin,0,device[d].CurrentStream>>>(device[d].DevSpins,
									  device[d].Red);
  }
  Reduce(12, NumberOfTemperatures, device, OrderParameters);
}


int main(int argc, char **argv)
{
  std::vector<dev> device;
  std::vector<double> ThetaSolid, Energy, CosThetaSolid, Temperatures;
  TYPE **Confs = NULL;
  int NumberOfTemperatures = 128;
  int NumberOfDevices = 1;
  int McSteps = 500000;
  vector<int> McStepsTable;
  vector <int> McStepsTable2;
  int McStepsSA = 10;
  int SimulatedAnnealingSteps = 50000;
  int McStepsPT = 500;
  int _LogScale = 0;
  int ParallelTemperingDuringMeasurement=0;
  int PTSteps = 500;
  int NumberOfRepetitions=1;
  int StoreConfigurations = 0;
  double tmin = 0.0005;
  double tmax = 1.0;
  dim3 n;
  int _Quadrupole = 0;
  int DChiDT = 0;
  int _OrderParameter = 0;
  int EveryMeasures= 100;
  double B[3] = {1,0,0};
  int MagneticField = 0;
  TYPE * scratch = NULL;
  int ContinueCalculation = 0;
  int _OverRelaxation = 0;
  int _Magnetization = 0;
  int NumberOfConfs=0;
  int NumberOfConfigurations = 1000;
  int ZAxisSum = 0;
  int q0=0;
  int _Test = 0;
  int ZAxis = 0;
  int _StructFactor = 0;
  int _SpinFlip = 0;
  int nsupcell=4;
  Lattice *lat = NULL;
  TYPE *OrderParameters = NULL;
  char *OrderParametersFile = NULL;
  char *MagnetizationFile = NULL;
  char *QuadrupoleFile = NULL;
  char *SzFile = NULL;
  char *SzSumFile = NULL;
  char *EnergyFile = NULL;
  char *EMFile = NULL;
  StructFactor *Sq = NULL;
  int _SaveTemperatures = 0;
  int SoA = 0;
  int NoEquilibration=0;
  fft *f1;
  n.x = 8;
  n.y = 8;
  n.z = 8;
  BlockSpin.x = 4;
  BlockSpin.y = 4;
  BlockSpin.z = 4;
  
  McSteps = 50000;
  int NumberOfMeasures = 100;
  int ThermalizationSteps = 50000;
  unsigned int ReduceAngle = 0;
  int _Projection = 0;
  struct poptOption GraphOptions[] = {
    {"nx", 'x', POPT_ARG_INT, &n.x, 0, "number of unit cell along x", "4"},
    {"ny", 'y', POPT_ARG_INT, &n.y, 0, "number of unit cell along y", "4"},
    {"nz", 'z', POPT_ARG_INT, &n.z, 0, "number of unit cell along z", "4"},
    {"bx", 0, POPT_ARG_INT, &BlockSpin.x, 0, "number of unit cell along x", "4"},
    {"by", 0, POPT_ARG_INT, &BlockSpin.y, 0, "number of unit cell along y", "4"},
    {"bz", 0, POPT_ARG_INT, &BlockSpin.z, 0, "number of unit cell along z", "4"},
    {"ReductionAngle", 0, POPT_ARG_NONE, &ReduceAngle, 0, "Apply angle reduction", "0"},
    {"McSteps", 'M', POPT_ARG_INT, &McSteps, 0, "Number of Montecarlo Steps", "50000"},
    {"OverRelaxation",0,POPT_ARG_INT,&_OverRelaxation, 0, "Number of over relaxation steps (indep. of the montecarlo)","0"},
    {"NumberOfDevices", 0, POPT_ARG_INT, &NumberOfDevices, 0, "Number of cuda card to use if possible", "1"},
    {"SimulatedAnnealingSteps", 'A', POPT_ARG_INT, &SimulatedAnnealingSteps, 0, "number of steps for the simulated Annealing (number of intermediate temperatures)", "50000"},
    {"McStepSimulatedAnnealing", 0, POPT_ARG_INT, &McStepsSA, 0, "number of MC steps between two simulated annealing steps", "10"},
    {"McStepsParallelTempering", 'P', POPT_ARG_INT, &McStepsPT, 0, "Number of Montecarlo Steps for the parallel Tempering", "500"},
    {"ThermalizationSteps", '\0', POPT_ARG_INT, &ThermalizationSteps, 0, "Number of Montecarlo Steps for the thermalization", "50000"},
    {"ParallelTemperingSteps", 'S', POPT_ARG_INT, &PTSteps, 0, "Number of Parallel tempering swaps", "500"},
    {"NumberOfMeasures",'N', POPT_ARG_INT, &NumberOfMeasures, 0, "Number of independent measures", "100"},
    {"NumberOfRepetitions",'r', POPT_ARG_INT, &NumberOfRepetitions, 0, "Number of independent repetitions", "1"},
    {"TMin", 0, POPT_ARG_DOUBLE, &tmin, 0, "Minimum temperature", "0.005"},
    {"TMax", 0, POPT_ARG_DOUBLE, &tmax, 0, "Max temperature", "1.0"},
    {"Zaxis", 0, POPT_ARG_NONE, &ZAxis, 0, " Compute the |Sz| for spin ice", "No"},
    {"ZaxisSum", 0, POPT_ARG_NONE, &ZAxisSum, 0, " Compute the |sum Sz|**2 for spin ice", "No"},
    {"NumberOfConfigurations", 0, POPT_ARG_INT, &NumberOfConfigurations, 0, "Number of snapshots", "1000"},
    {"LogScale", 0, POPT_ARG_NONE, &_LogScale, 0, "log scale temperatures", "0"},
    {"OrderParameters",0,POPT_ARG_NONE, &_OrderParameter, 0, "Compute the order parameter","0"},
    {"Quadrupole",0,POPT_ARG_NONE, &_Quadrupole, 0, "Compute the quadrupolar order parameter","0"},
    {"StoreConfigurations",0,POPT_ARG_NONE,&StoreConfigurations,0,"Store the Configurations every 500 Mc steps", "0"},
    {"NumberOfTemperatures", 0, POPT_ARG_INT, &NumberOfTemperatures, 0, "Number of temperatures", "20"},
    {"ApplyParallelTemperingDuringMeasure", 0, POPT_ARG_NONE, &ParallelTemperingDuringMeasurement, 0, "Apply the Parallel tempering during Measurement", "No"},
    {"NoEquilibration", 0, POPT_ARG_NONE, &NoEquilibration, 0, "Start measurement using previously stored configurations", "0"},
    {"SoA", 0, POPT_ARG_NONE, &SoA,0,"Convert Array of structure to struct of array. The AoS was used so far. Only useful for reading configurations generated before Oct. 2016", "0"},
    {"SaveTemperatures", 0, POPT_ARG_NONE, &_SaveTemperatures, 0, "Save temperature scale", "0"},
    {"Test",0,POPT_ARG_NONE, &_Test, 0, "Testing purpose only just run one mc step", "0"},
    {"ContinueMeasurement",0,POPT_ARG_NONE, &ContinueCalculation, 0, "Continue measurement after stoping the code", "0"},
    {"Magnetization",0,POPT_ARG_NONE, &_Magnetization, 0, "Compute the magnetization with g tensor", "0"},
    {"ConfEveryMeasures", 0, POPT_ARG_INT, &EveryMeasures, 0, "Take a configuration every 100 measures for instance", "100"},
    {"StructFactor", 0, POPT_ARG_NONE, &_StructFactor, 0, "Calculate the structure factor", "0"},
    {"SpinFlip", 0, POPT_ARG_NONE, &_SpinFlip, 0, "Calculate the structure factor SpinFlip non SpinFlip", "0"},
    {"ZB", 0, POPT_ARG_INT, &nsupcell, 0, "Number of brillouin zones", "4"},
    {"Projection", 0, POPT_ARG_NONE, &_Projection, 0, "Calculate the projected structure factor ", "0"},
    {"DChiDt", 0, POPT_ARG_NONE, &DChiDT, 0, "Compute the temperature derivative of chi", "0"},
    {"Rng",'R', POPT_ARG_INT, &RngSeed, 0, "Initial Seed for the Rng", "56081376"},
    {"J1", 0, POPT_ARG_DOUBLE, &J1, 0, "value of J1 (default YbTiO)", "0.31034"},
    {"J2", 0, POPT_ARG_DOUBLE, &J2, 0, "value of J2 (default YbTiO)", "0.75862"},
    {"J3", 0, POPT_ARG_DOUBLE, &J3, 0, "value of J3 (default YbTiO)", "1"},
    {"J4", 0, POPT_ARG_DOUBLE, &J4, 0, "value of j4 (default YbTiO)", "0"},
    {"MagneticField", 0, POPT_ARG_NONE, &MagneticField, 0, "Add the magnetic field", "off"},
    {"Bx", 0, POPT_ARG_DOUBLE, &B[0], 0, "coordinates of the magnetic field along x (include the amplitude in Tesla)", "1"},
    {"By", 0, POPT_ARG_DOUBLE, &B[1], 0, "coordinates of the magnetic field along y", "0"},
    {"Bz", 0, POPT_ARG_DOUBLE, &B[2], 0, "coordinates of the magnetic field along z", "0"},
    {"gxy", 0, POPT_ARG_DOUBLE, &gxy, 0, "g-tensor gxy", "1.0"},
    {"gz", 0, POPT_ARG_DOUBLE, &gz, 0, "g-tensor gz", "1.0"},
    {"q0", 0, POPT_ARG_NONE, &q0, 0, "Generate thermalized confs for $q=0$ state", "0"},
    POPT_AUTOHELP
    POPT_TABLEEND
  };
  
  poptContext optCon = NULL;
  optCon = poptGetContext(argv[0], argc, (const char **)argv, GraphOptions, 0);
  int error_option = 1;
  if(argc == 1)
    {
      poptPrintUsage(optCon, stdout, 0);
      poptFreeContext(optCon);
      exit(0);
    }
  
  while (error_option) {
    error_option = poptGetNextOpt(optCon);
    if(error_option==-1) break;
    switch(error_option) {
    case POPT_ERROR_BADOPT:
      fprintf(stderr, "%s: %s. See --help for more details.\n",
	      poptBadOption(optCon, POPT_BADOPTION_NOALIAS),
	      poptStrerror(error_option));
      exit(1);
      break;
    case POPT_ERROR_NOARG:
      fprintf(stderr, "%s\n. See the --help option for more details\n", poptStrerror(error_option));
      exit(1);
      break;
    default:
      break;
    }
  }
  int nspins = NumberOfSpinsPerUnitCell * n.x * n.y * n.z;
  int ntmp[3] = {n.x, n.y, n.z};
  
  vslNewStream(&RngStream, VSL_BRNG_MT19937, RngSeed);
  
  InitializeDevice(ntmp,
		   NumberOfSpinsPerUnitCell,
		   NumberOfTemperatures,
		   RngSeed,
		   3,
		   device); // Each gpu treat 8 temperatures
  NumberOfDevices = device.size();
  // I do not use shared memory for this model because it is slower than without
  for(auto d=0;d<device.size();d++)
    device[d].SharedMemory = 0; //6*sizeof(TYPE3)*BlockSpin.x*BlockSpin.y*BlockSpin.z;
  if(device.size()==0)
    {
      printf("Something is wrong\n");
      exit(1);
    }
  
  InitializeCouplings3D(ntmp, NumberOfSpinsPerUnitCell, SpinPerPrimitiveCell, NumberOfNeighbors, MagneticField,  B, device);
  cudaSetDevice(0);
  std::vector <size_t> ConfigurationsPermutation(NumberOfTemperatures);
  InitializeTables(&Confs,
		   Energy,
		   CosThetaSolid,
		   ThetaSolid,
		   McStepsTable,
		   Temperatures,
		   NumberOfTemperatures,
		   nspins);
  scratch = (TYPE *)malloc(sizeof(TYPE)*NumberOfTemperatures*32);

  for(auto s=0;s<nspins;s++) {
    Confs[0][s]=1;
    Confs[0][s+nspins]=0;
    Confs[0][s+2*nspins]=0;
  }
  
  printf("Total energy %.10lf\n", TotalEnergy(Confs[0],n));
  
  if(_LogScale)
    SetupLogTemperatures(Temperatures, tmax, tmin, 1.0/NumberOfTemperatures);
  else
    SetupLinearTemperatures(Temperatures, tmax, tmax/((double)NumberOfTemperatures));

  if(_SaveTemperatures)
    SaveTemperatures(Temperatures);
    
  EnergyFile = (char *)malloc(sizeof(char)*128);
  SzFile = (char *)malloc(sizeof(char)*128);
  SzSumFile = (char *)malloc(sizeof(char)*128);
  QuadrupoleFile = (char *)malloc(sizeof(char)*128);
  OrderParametersFile = (char *)malloc(sizeof(char)*128);
  MagnetizationFile = (char *)malloc(sizeof(char)*128);
  EMFile = (char *)malloc(sizeof(char)*128);
  if(_StructFactor) {
    vector <vector <double> > atoms;
    vector <vector<double> > basis(3);
    basis[0].clear();
    basis[1].clear();
    basis[2].clear();
    basis[0].resize(3);
    basis[1].resize(3);
    basis[2].resize(3);
    
    basis[0][0] = 1.0;
    basis[0][1] = 0.0;
    basis[0][2] = 0.0;
    basis[1][0] = 0.0;
    basis[1][1] = 1.0;
    basis[1][2] = 0.0;
    basis[2][0] = 0.0;
    basis[2][1] = 0.0;
    basis[2][2] = 1.0;
    vector <int> size(3);
    size[0] = n.x;
    size[1] = n.y;
    size[2] = n.z;
    vector<int> supcell(3);
    supcell[0] = nsupcell;
    supcell[1] = nsupcell;
    supcell[2] = nsupcell;
    
    atoms.clear();
    atoms.resize(16);
    for(auto i=0;i<16;i++) {
      atoms[i].clear();
      atoms[i].resize(3);
    }
    
    
    atoms[0][0]=0;
    atoms[0][1]=0;
    atoms[0][2]=0;
    
    atoms[1][0]=0.250000000000000;
    atoms[1][1]=0.250000000000000;
    atoms[1][2]=0;
    
    atoms[2][0]=0;
    atoms[2][1]=0.250000000000000;
    atoms[2][2] = 0.250000000000000;
    
    atoms[3][0]=0.250000000000000;
    atoms[3][1]=0;
    atoms[3][2]=0.250000000000000;
    
    atoms[4][0]=0.500000000000000;
    atoms[4][1]=0.500000000000000;
    atoms[4][2]=0;
    
    atoms[5][0]=0.750000000000000;
    atoms[5][1]=0.750000000000000;
    atoms[5][2]=0;
    
    atoms[6][0]=0.500000000000000;
    atoms[6][1]=0.750000000000000;
    atoms[6][2]=0.250000000000000;
    
    atoms[7][0]=0.750000000000000;
    atoms[7][1]=0.500000000000000;
    atoms[7][2]=0.250000000000000;
    
    atoms[8][0]=0;
    atoms[8][1]=0.500000000000000;
    atoms[8][2]=0.500000000000000;
    
    atoms[9][0]=0.250000000000000;
    atoms[9][1]=0.750000000000000;
    atoms[9][2]=0.500000000000000;
    
    atoms[10][0]=0;
    atoms[10][1]=0.750000000000000;
    atoms[10][2]=0.750000000000000;
    
    atoms[11][0]=0.250000000000000;
    atoms[11][1]=0.500000000000000;
    atoms[11][2]=0.750000000000000;
    
    atoms[12][0]=0.500000000000000;
    atoms[12][1]=0;
    atoms[12][2]=0.500000000000000;
    
    atoms[13][0]=0.750000000000000;
    atoms[13][1]=0.250000000000000;
    atoms[13][2]=0.500000000000000;
    
    atoms[14][0]=0.500000000000000;
    atoms[14][1]=0.250000000000000;
    atoms[14][2]=0.750000000000000;
    
    atoms[15][0]=0.750000000000000;
    atoms[15][1]=0;
    atoms[15][2]=0.750000000000000;
    
    
    lat   = new Lattice(supcell, basis, atoms, size);
    lat->GenerateFourierCoefficients();
    f1 = new fft(*lat, 3, device);
    Sq = new StructFactor(*lat,
			  *f1,
			  NumberOfTemperatures,
			  "Sq",
			  0,
			  0,
			  _SpinFlip,
                          _Projection);
    supcell.clear();
    basis.clear();
    atoms.clear();
    size.clear();
  }

  vector <double> TemperaturesNext(NumberOfTemperatures);
  for(auto T=0;T<TemperaturesNext.size();T++)
    TemperaturesNext[T] = Temperatures[T];
  
  if((!_Test)&&(!q0)) {
    Measure EnergyMeasure(NumberOfTemperatures*2, TemperaturesNext);
    Measure MagnetizationMeasure(NumberOfTemperatures, TemperaturesNext);
    Measure Quadrupole(3*NumberOfTemperatures, TemperaturesNext);
    Measure OrderParametersMeasure(10*NumberOfTemperatures, TemperaturesNext);
    Measure Sz(NumberOfTemperatures, TemperaturesNext);
    Measure SzSum(NumberOfTemperatures, TemperaturesNext);
    Measure EM(10*NumberOfTemperatures, TemperaturesNext); // compute M E and M^2 E. Useful for the calcualtions of d chi /d T
    EnergyMeasure.Reset();
    MagnetizationMeasure.Reset();
    Quadrupole.Reset();
    OrderParametersMeasure.Reset();
    SzSum.Reset();
    EM.Reset();


    for(auto rep=0;rep<NumberOfRepetitions;rep++) {

      if(!NoEquilibration) {
	printf("Configuration Series %d\n", rep);
	
	if(_LogScale)
	  SetupLogTemperatures(Temperatures, tmax, tmin, 1.0/NumberOfTemperatures);
	else
	  SetupLinearTemperatures(Temperatures, tmax, tmax/((double)NumberOfTemperatures));
	
	if(_SaveTemperatures) {
	  SaveTemperatures(Temperatures);
	  _SaveTemperatures = 1;
	}
	
	for(auto d=0;d<NumberOfTemperatures;d++) {
	  McStepsTable[d] = McStepsSA;
	}
	
	SimulatedAnnealing(device,
			   1,
			   SimulatedAnnealingSteps,
			   _OverRelaxation,
			   ReduceAngle,
			   McStepsTable,
			   Temperatures,
			   20.0, // Starting temperature for the simulated annealing
			   CosThetaSolid);
	
	
	
	
	for(auto T=0;T<NumberOfTemperatures;T++) {
	  McStepsTable[T] = McStepsPT;
	}
	
	printf("Starting the parallel tempering\n");
	for(auto run=0;run<PTSteps;run++) {
	  for(auto T=0;T<NumberOfTemperatures;T++) {
	    McStepsTable[T] = McStepsPT;
	  }
	  
	  ParallelTempering(device,
			    _OverRelaxation,
			    ReduceAngle,
			    McStepsTable,
			    run%2,
			    Temperatures,
			    CosThetaSolid,
			    Energy);
	}
	
	for(auto T=0;T<NumberOfTemperatures;T++) {
	  McStepsTable[T] = ThermalizationSteps;
	}
	
	printf("Starting Thermalization\n");
	MonteCarlo(device,
		   _OverRelaxation,
		   ReduceAngle,
		   McStepsTable,
		   Temperatures,
		   CosThetaSolid,
		   NULL);

	SynchronizeJobs(device);
	ExtractConfigurationsFromMonteCarlo(device);
	gsl_sort_largest_index (&ConfigurationsPermutation[0], 
				NumberOfTemperatures, 
				&Temperatures[0], 
				1, 
				NumberOfTemperatures);
	TotalEnergyv2(device,
		      Energy);
	CopyResultsOnCPU(Confs, device);

	SaveConfigurations<double>((const TYPE **)Confs, 
				   Temperatures,
				   nspins,
				   3,
				   "InitialConf",
				   rep);
	
	for(auto d=0;d<NumberOfTemperatures;d++) {
	  printf("%.10lf %.10lf %.10lf\n", Temperatures[d], Energy[d], TotalEnergy(Confs[d], n));
	}
	printf("\n");
      } else {
	ReadTemperatures(Temperatures);
	ReadConfigurations(Confs, NumberOfTemperatures, nspins, 3, "InitialConf", rep);
	CopyResultsOnGPU(Confs, device);
	printf("reading Confs and calculate energy\n");
	
	/* for(auto d=0;d<NumberOfTemperatures;d++) { */
	/*   printf("%.10lf  %.10lf\n", Temperatures[d], TotalEnergy(Confs[d], n)); */
	/* } */
	printf("Starting Thermalization\n");
		
	for(auto T=0;T<NumberOfTemperatures;T++) {
	  McStepsTable[T] = ThermalizationSteps;
	}

	MonteCarlo(device,
		   _OverRelaxation,
		   ReduceAngle,
		   McStepsTable,
		   Temperatures,
		   CosThetaSolid,
		   NULL);

	//	SynchronizeJobs(device);
      }
      
      /* CopyResultsOnCPU(Confs, device); */
      /* for(auto d=0;d<NumberOfTemperatures;d++) { */
      /* 	printf("%.10lf  %.10lf\n", Temperatures[d], TotalEnergy(Confs[d], n)); */
      /* } */
      
      if(NumberOfMeasures) {
	printf("Starting measurement\n");
	
	for(auto T=0;T<NumberOfTemperatures;T++) {
	  McStepsTable[T] = McSteps/NumberOfMeasures;
	}
	memset(EnergyFile, 0, sizeof(char)*128);
	sprintf(EnergyFile,"Energies%d.dat", rep);
	
	OrderParameters = (double *)malloc(sizeof(double)*12*NumberOfTemperatures);

	if(DChiDT) {
	  _OrderParameter=1;
	  memset(EMFile, 0, 128);
	  sprintf(EMFile, "dchidt%d.dat", rep);
	}
	
	if(_OrderParameter) {
	  memset(OrderParametersFile, 0, sizeof(char)*128);
	  sprintf(OrderParametersFile, "OrderParameters%d.dat", rep);
	}
	
	if(ZAxis) {
	  memset(SzFile,0,128);
	  sprintf(SzFile, "Sz%d.dat", rep);
	}


	if(ZAxisSum) {
	  memset(SzSumFile, 0, sizeof(char)*128);
	  sprintf(SzSumFile, "SzSum%d.dat", rep);
	}
	
	if(_Quadrupole) {
	  memset(QuadrupoleFile, 0, sizeof(char)*128);
	  sprintf(QuadrupoleFile, "Quadrupole%d.dat", rep);
	}
	
	if(_Magnetization) {
	  memset(MagnetizationFile, 0, sizeof(char)*128);	
	  sprintf(MagnetizationFile,"Magnetization%d.dat", rep);
	}
	
	int ddd=0;
	
	for(auto m=0;m<NumberOfMeasures;m++) {
	  if(!(m%100))
	    printf("Measure %d done\n",m);

	  // This makes a copy of the configruation in a working table
	  // so that we can run the montecarlo in parallel of during
	  // measurement
	  
	  // Needs to synchronize the threads

	  SynchronizeJobs(device);

	  // This should be not blocking
	  ExtractConfigurationsFromMonteCarlo(device);
	  
	  gsl_sort_largest_index (&ConfigurationsPermutation[0], 
				  NumberOfTemperatures, 
				  &Temperatures[0], 
				  1, 
				  NumberOfTemperatures);
	  
	  // Copy the temperatures
	  for(auto T=0;T<TemperaturesNext.size();T++)
	    TemperaturesNext[T] = Temperatures[T];
	  
	  int Apply = (((m*McStepsTable[0]) % McStepsPT) == 0)&&(ParallelTemperingDuringMeasurement);

	  if(ApplyParallelTemperingDuringMeasure(Apply+ddd,
						 _OverRelaxation, 
						 ReduceAngle, 
						 McStepsTable, 
						 CosThetaSolid, 
						 TemperaturesNext, 
						 device))
	    ddd++;
	  
	  // It is the only part that is synchronized. All the rest
	  // including the next montecarlo is async.	  

	  TotalEnergyv2(device, Energy); // there is explicit synchronization in totalenergy

	  for(auto d=0;d<NumberOfTemperatures;d++) {
	    scratch[2*d] = Energy[ConfigurationsPermutation[d]];
	    scratch[2*d+1] = Energy[ConfigurationsPermutation[d]]*Energy[ConfigurationsPermutation[d]];
	  }	  
	  
	  EnergyMeasure.Accumulate(scratch);

	  if(_OrderParameter) {
	    DoMeasurement(n.x*n.y*n.z, NumberOfTemperatures, device, OrderParameters);
	    for(auto T=0;T<NumberOfTemperatures;T++) {
	      TYPE *m1 = &OrderParameters[12*ConfigurationsPermutation[T]];
	      TYPE *m2 = &OrderParameters[12*ConfigurationsPermutation[T]+1];
	      TYPE *m3 = &OrderParameters[12*ConfigurationsPermutation[T]+3];
	      TYPE *m4 = &OrderParameters[12*ConfigurationsPermutation[T]+6];
	      TYPE *m5 = &OrderParameters[12*ConfigurationsPermutation[T]+9];
	      scratch[10*T] = sqrt(m1[0]*m1[0]);
	      scratch[10*T+1] = sqrt(m2[0]*m2[0] + m2[1]*m2[1]);
	      scratch[10*T+2] = sqrt(m3[0]*m3[0] + m3[1]*m3[1] + m3[2]*m3[2]);
	      scratch[10*T+3] = sqrt(m4[0]*m4[0] + m4[1]*m4[1] + m4[2]*m4[2]);
	      scratch[10*T+4] = sqrt(m5[0]*m5[0] + m5[1]*m5[1] + m5[2]*m5[2]);
	      scratch[10*T+5] = m1[0]*m1[0];
	      scratch[10*T+6] = m2[0]*m2[0] + m2[1]*m2[1];
	      scratch[10*T+7] = m3[0]*m3[0] + m3[1]*m3[1] + m3[2]*m3[2];
	      scratch[10*T+8] = m4[0]*m4[0] + m4[1]*m4[1] + m4[2]*m4[2];
	      scratch[10*T+9] = m5[0]*m5[0] + m5[1]*m5[1] + m5[2]*m5[2];
	    }
	    OrderParametersMeasure.Accumulate(scratch);
	    // now calculate variance, mean, etc...
	  }

	  if(DChiDT) {
	    for(auto T=0;T<NumberOfTemperatures;T++) {
	      for(auto ord=0;ord<10;ord++) {
		scratch[10*T+ord] *= Energy[ConfigurationsPermutation[T]];
	      }
	    }
	    EM.Accumulate(scratch);
	  }
	  //printf("Mc\n");
	  // run the montecarlo while doing measures
	  
	  //printf("Mc2\n");
	  
	  if(_StructFactor) {          
	    // Data are already ready

	    Sq->CalculateStructureFactor(*lat, *f1, ConfigurationsPermutation, device);
	  }
	  
	  if(ZAxis) {
	    CalculateAbsSz(n.x*n.y*n.z, NumberOfTemperatures, device, OrderParameters);
	    for(auto d=0;d<NumberOfTemperatures;d++)
	      scratch[d] = OrderParameters[ConfigurationsPermutation[d]];
	    Sz.Accumulate(scratch);
	  }
	  
	  if(ZAxisSum) {
	    CalculateAbsSumSz(n.x*n.y*n.z, NumberOfTemperatures, device, OrderParameters);
	    for(auto d=0;d<NumberOfTemperatures;d++)
	      scratch[d] = OrderParameters[ConfigurationsPermutation[d]];
	    SzSum.Accumulate(scratch);
	  }
	  
	  
	  if(_Quadrupole) {
	    CalculateQuadrupole(NumberOfTemperatures, device, OrderParameters);
	    for(auto T=0;T<NumberOfTemperatures;T++) {
	      double qxx, qyy, qxy, qxz, qyz, qzz;
	      qxx = OrderParameters[6*ConfigurationsPermutation[T]];
	      qxy = OrderParameters[6*ConfigurationsPermutation[T]+1];
	      qxz = OrderParameters[6*ConfigurationsPermutation[T]+2];
	      qyy = OrderParameters[6*ConfigurationsPermutation[T]+3];
	      qyz = OrderParameters[6*ConfigurationsPermutation[T]+4];
	      qzz = OrderParameters[6*ConfigurationsPermutation[T]+5];
	      //printf("%.5lf %.5lf %.5lf %.5lf %.5lf\n", scratch[5*T], scratch[5*T+1], scratch[5*T+2], scratch[5*T+3], scratch[5*T+4]);
	      scratch[3*T] = sqrt((2.0*qxx - qyy -qzz)*(2.0*qxx - qyy -qzz) + 3.0*(qzz-qyy)* (qzz-qyy));
	      scratch[3*T+1] = sqrt(qxy*qxy + qxz*qxz + qyz*qyz);
	      scratch[3*T+2] = sqrt(qxx*qxx + qyy*qyy + qzz*qzz +qxy*qxy +qxz*qxz +qyz*qyz);
	    }
	    
	    Quadrupole.Accumulate(scratch);
	  }
	  
	  if(_Magnetization) {
	    Magnetization(NumberOfTemperatures,
			  device,
			  scratch + NumberOfTemperatures);
	    for(auto d=0;d<NumberOfTemperatures;d++) {
	      TYPE *m = scratch + NumberOfTemperatures;
	      int TT= ConfigurationsPermutation[d];
	      scratch[d] = sqrt(m[3*TT] * m[3*TT] +
				m[3*TT+1] * m[3*TT+1] +
				m[3*TT+2] * m[3*TT+2]);
	    }
	    MagnetizationMeasure.Accumulate(scratch);
	  }
	  
	  if((m%500)==0) {
	    for(auto T=0;T<NumberOfTemperatures;T++) {
	      scratch[T] = Temperatures[ConfigurationsPermutation[T]];
	    }
	    
	    EnergyMeasure.SaveData(EnergyFile, NumberOfTemperatures, 2);
	    if(_Quadrupole) 
	      Quadrupole.SaveData(QuadrupoleFile, NumberOfTemperatures, 3);
	    if(_OrderParameter)
	      OrderParametersMeasure.SaveData(OrderParametersFile, NumberOfTemperatures, 10);
	    if(ZAxisSum)
	      SzSum.SaveData(SzSumFile, NumberOfTemperatures, 1);
	    if(ZAxis)
	      Sz.SaveData(SzFile, NumberOfTemperatures, 1);
	    if(_Magnetization)
	      MagnetizationMeasure.SaveData(MagnetizationFile, NumberOfTemperatures, 1);
	    if(DChiDT)
	      EM.SaveData(EMFile, NumberOfTemperatures, 10);
	    if(_StructFactor) {
	      Sq->CreateStructFactorFile(*lat, NumberOfMeasures, NumberOfTemperatures, "StructFactor");
	    }
	  }
	  
	  
	  if((m%(EveryMeasures)==0)&&(StoreConfigurations)&&(NumberOfConfs<NumberOfConfigurations)) {
	    printf("Confs Series %d\n", m);
	    // The configurations saved here are the one used for the measurement. not the next one.
	    CopyResultsOnCPU(Confs, device);
	    SaveConfigurations<double>((const double **)Confs, Temperatures, nspins, 3, "Conf", 1);
	    NumberOfConfs++;
	  }	  
	}
	
	for(auto T=0;T<NumberOfTemperatures;T++) {
	  scratch[T] = Temperatures[ConfigurationsPermutation[T]];
        }
	
	EnergyMeasure.SaveData(EnergyFile, NumberOfTemperatures, 2);
	
	if(_Quadrupole) 
	  Quadrupole.SaveData(QuadrupoleFile, NumberOfTemperatures, 3);
	if(_OrderParameter)
	  OrderParametersMeasure.SaveData(OrderParametersFile, NumberOfTemperatures, 10);
	if(ZAxisSum)
	  SzSum.SaveData(SzSumFile, NumberOfTemperatures, 1);
	if(ZAxis)
	  Sz.SaveData(SzFile, NumberOfTemperatures, 1);
	if(_Magnetization)
	  MagnetizationMeasure.SaveData(MagnetizationFile, NumberOfTemperatures, 1);
	if(DChiDT)
	  EM.SaveData(EMFile, NumberOfTemperatures, 10);
	if(_StructFactor) {
	  Sq->CreateStructFactorFile(*lat, NumberOfMeasures, NumberOfTemperatures, "StructFactor");
	}
	EnergyMeasure.Reset();
	MagnetizationMeasure.Reset();
	Quadrupole.Reset();
	OrderParametersMeasure.Reset();
	Sz.Reset();
	SzSum.Reset();
	EM.Reset();
      }

      if(NumberOfRepetitions > 1)
	SetNewGenerator(device);

    }
  } else {
    GenerateRandomVectorsOnDevice(device);
    for(auto T=0;T<NumberOfTemperatures;T++) {
      McStepsTable[T] = 1000;
    }
    MonteCarlo(device,
    	       _OverRelaxation,
    	       ReduceAngle,
    	       McStepsTable,
    	       Temperatures,
    	       CosThetaSolid,
    	       NULL);
    SynchronizeJobs(device);
    ExtractConfigurationsFromMonteCarlo(device);
    TotalEnergyv2(device, Energy);
    printf("%.5lf\n", -Energy[0]);

    /* for(auto T=0;T<NumberOfTemperatures;T++) { */
    /*   McStepsTable[T] = 1000; */
    /* } */
    printf("testing\n");
    MonteCarlo(device,
    	       _OverRelaxation,
    	       ReduceAngle,
    	       McStepsTable,
    	       Temperatures,
    	       CosThetaSolid,
    	       NULL);

    ExtractConfigurationsFromMonteCarlo(device);
    TotalEnergyv2(device, Energy);
    printf("%.5lf\n", -Energy[0]);
  }

  if(lat) {
    delete lat;
    delete f1;
    delete Sq;
  }
  DestroyDevice(device);
  ConfigurationsPermutation.clear();
  Energy.clear();
  ThetaSolid.clear();
  CosThetaSolid.clear();
  Temperatures.clear();
  McStepsTable.clear();
  free(Confs[0]);
  free(Confs);
  free(SzSumFile);
  free(SzFile);
  free(MagnetizationFile);
  free(OrderParametersFile);
  free(QuadrupoleFile);
  free(EnergyFile);
  free(EMFile);
  free(scratch);
  return EXIT_SUCCESS;
}
