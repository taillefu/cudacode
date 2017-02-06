#include "functions.h"
#include <vector>
#include <typeinfo>
__constant__ int __PrimitiveCells[1];
__constant__ int __SpinPerPrimitiveCell[1];
__constant__ int __SystemSize[3]; 
__constant__ int __links[16*6]; 
__constant__ int __NumberOfSpinsPerUnitCell[1] ;
__constant__ int __NumberOfNeighbors[24];
__constant__ int __NumberOfLinks[1];
__constant__ int __BlocksPerTemperature[3];
__constant__ int __TemperatureGrid[3];
__constant__ int __IndependentSites[NumberOfParallelUpdates];
__constant__ int __PeriodicBoundariesConditions[1];

dim3 BlockSpin;

extern void SynchronizeJobs(vector<dev> &device);

extern GLOBAL void TotalEnergyGPU(const TYPE_SPIN *spin, TYPE *Energy);
extern  double TotalEnergy(const TYPE_SPIN *spin, const dim3 n);

void CalculateRatio(const int NumberOfTemperatures, const int McSteps, std::vector<dev> &device, TYPE *Ratio);
void TotalEnergyv2(const int NumberOfTemperatures, std::vector<dev> &device, double *Energy);
__global__ void Generate3DVectorList(const TYPE *__restrict__ rng, const int n, TYPE *__restrict__ spin);
void CopyResultsOnCPU(TYPE_SPIN **Confs, std::vector<dev> &device);
void CopyResultsOnGPU(TYPE_SPIN **Confs, std::vector<dev> &device);
#include "reduce.cu"

// ---------------------------------------------------------------------------------------- 
// Setup different constants for calculation on gpus

__device__ __inline__ void   InitializeConstants2D(int3 *n, 
						   int *blk, 
						   int *TemperatureNumber, 
						   int2 *thd, 
						   int *tht, 
						   int *LocalCell,
						   int *Cell, 
						   int *ncell, 
						   int *Offset)
{
  *TemperatureNumber = blockIdx.z;
  if(blk)
    *blk = (blockIdx.y)*gridDim.x + blockIdx.x;
  
  n->z = __NumberOfSpinsPerUnitCell[0];
  n->x = __SystemSize[0];
  n->y = __SystemSize[1];
  if(LocalCell)
    *LocalCell = threadIdx.x + threadIdx.y*blockDim.x;
  if(thd != NULL) {
    thd->x = threadIdx.x + blockDim.x*blockIdx.x;
    thd->y = threadIdx.y + blockDim.y*blockIdx.y;
  }
  if(tht != NULL)
    *tht = blockDim.x*blockDim.y;
  if(Cell != NULL)
    *Cell = thd->y*n->x+thd->x;
  if(ncell)
    *ncell = n->x*n->y;

#if !defined(DISCRETE)
  *Offset = TemperatureNumber[0]*ncell[0]*n->z;
#else
  *Offset = TemperatureNumber[0]*ncell[0];
#endif
} 

__device__ __inline__ void   InitializeConstants3D(int4 *n, int *blk, int *TemperatureNumber, int3 *thd, int *tht, int *LocalCell, int *Cell, int *ncell, int *Offset)
{
  n->x = blockIdx.x/__BlocksPerTemperature[0];
  n->y = blockIdx.y/__BlocksPerTemperature[1];
  n->z = blockIdx.z/__BlocksPerTemperature[2];  
  *TemperatureNumber = (n->z*__TemperatureGrid[1]+n->y)*__TemperatureGrid[0] + n->x;
  n->x = blockIdx.x%__BlocksPerTemperature[0];
  n->y = blockIdx.y%__BlocksPerTemperature[1];
  n->z = blockIdx.z%__BlocksPerTemperature[2];  

  *blk = (n->z * __BlocksPerTemperature[1] + n->y)* __BlocksPerTemperature[0] + n->x;
  n->w = __NumberOfSpinsPerUnitCell[0];
  n->x = __SystemSize[0];
  n->y = __SystemSize[1];
  n->z = __SystemSize[2];
  thd->x = threadIdx.x + blockDim.x*(blockIdx.x%__BlocksPerTemperature[0]);
  thd->y = threadIdx.y + blockDim.y*(blockIdx.y%__BlocksPerTemperature[1]);
  thd->z = threadIdx.z + blockDim.z*(blockIdx.z%__BlocksPerTemperature[2]);
  *tht = blockDim.x*blockDim.y*blockDim.z;
  *LocalCell = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x+threadIdx.x;
  *Cell = (thd->z*n->y + thd->y)*n->x+thd->x;
  *ncell = n->x*n->y*n->z;
#if (!defined(POTTS) && !defined(ISING))
  *Offset = TemperatureNumber[0]*ncell[0]*n->w;
#else
  *Offset = TemperatureNumber[0]*ncell[0];
#endif
} 




// -------------------------------------------------------------------------------------------------------
// Math operations on vectors

__device__ void CrossProduct(const TYPE3 *__restrict__ a, const TYPE3 * __restrict__ b, TYPE3 *c)
{
  TYPE ax = a->x;
  TYPE ay = a->y;
  TYPE az = a->z;
  
  TYPE bx = b->x;
  TYPE by = b->y;
  TYPE bz = b->z;

  c->x = ay * bz - az * by;
  c->y = bx * az - bz * ax;
  c->z = ax * by - bx * ay;
}

__device__ __host__ void __Generate3DVector(const TYPE rng1, const TYPE rng2, TYPE *__restrict__ spin)
{
  TYPE pc, ps, ts, tc;
  SINCOSPI(TWO*rng1, &ps, &pc);
  tc = TWO*rng2-ONE;
  ts = SQRT(1.0-tc*tc);
  
  spin[0] = pc*ts;
  spin[1] = ps*ts;
  spin[2] = tc;
}

__global__ void Generate3DVectorList(const TYPE *__restrict__ rng, const int n, TYPE *__restrict__ spin)
{
  int Id = threadIdx.x + blockIdx.x * blockDim.x;
  //TYPE *v = (TYPE *)shared_mem;
  if(Id >= n)
    return;
  TYPE v[3];
  __Generate3DVector(rng[Id], rng[Id+blockDim.x*gridDim.x], v);
  spin[3*Id] = v[0];
  spin[3*Id+1] = v[1];
  spin[3*Id+2] = v[2];
}

__global__ void GeneratePottsStates(const TYPE *__restrict__ rng, const int n, int *__restrict__ spin)
{
  int cell = threadIdx.x + blockIdx.x * blockDim.x;
  //TYPE *v = (TYPE *)shared_mem;
  if(cell >= n)
    return;
  int Cell = 0;
  for(int i=0;i<__NumberOfSpinsPerUnitCell[0];i++) {
    int state = (rng[__NumberOfSpinsPerUnitCell[0]*cell+i] >= 0.3333333333f) + (rng[__NumberOfSpinsPerUnitCell[0]*cell+i] >= 0.6666666666666f);
    Cell = (state << (2*i)) ^ Cell;
  }
  spin[cell] = Cell;
}

TYPE nrm(const int l, const TYPE *s)
{
  TYPE result = 0;
  for(int i=0;i<l;i++)
    result += s[i]*s[i];
  
  return result;
}

__device__ __host__ void _CudaRotate3DVector(const TYPE angle, const TYPE3 b, TYPE3 *__restrict__ a)
{
  // using quaternions for the rotation
  // q = n * sin (theta/2)
  // v' = v + 2 q x (qxv + cos(theta/2) v)
  
  const TYPE vx = a->x;
  const TYPE vy = a->y;
  const TYPE vz = a->z;
  
  TYPE ct;//COS(angle*0.5);
  TYPE st;//SIN(angle*0.5);
  SINCOSPI(angle*HALF, &st, &ct);

  const TYPE normN = st*RSQRT(b.x*b.x +
			      b.y*b.y +
			      b.z*b.z);
  
  const TYPE qx = b.x*normN;
  const TYPE qy = b.y*normN;
  const TYPE qz = b.z*normN;
  
  const TYPE tx = qy*vz-qz*vy + ct*vx;
  const TYPE ty = qz*vx-qx*vz + ct*vy;
  const TYPE tz = qx*vy-qy*vx + ct*vz;
  
  a->x = vx + TWO*(qy*tz-qz*ty);
  a->y = vy + TWO*(qz*tx-qx*tz); 
  a->z = vz + TWO*(qx*ty-qy*tx);
}

__global__  void _vzmulgpu(const int length, const TYPE *a, const TYPE *b, TYPE *c)
{
  if(threadIdx.x+blockIdx.x*blockDim.x>=length)
    return;
    
  const TYPE2 *a2 = (const TYPE2 *)a;
  const TYPE2 *b2 = (const TYPE2 *)b;
  TYPE2 a22, b22;
  a22 = a2[blockIdx.x*blockDim.x + threadIdx.x];
  b22 = b2[blockIdx.x*blockDim.x + threadIdx.x];
  __syncthreads();
  
  TYPE cc[2];
  cc[0] = a22.x * b22.x - a22.y * b22.y;
  cc[1] = a22.y * b22.x + a22.x * b22.y;
  
  c[2*(threadIdx.x+blockIdx.x*blockDim.x)] = cc[0];
  c[2*(threadIdx.x+blockIdx.x*blockDim.x)+1] = cc[1];
}


__device__ __inline__ __host__ void _CudaRotate3DVectorByPi(const TYPE3 b, TYPE3 *__restrict__ a)
{
  // using quaternions for the rotation
  // q = n * sin (theta/2)
  // v' = v + 2 q x (qxv + cos(theta/2) v)
  
  TYPE vx = a->x;
  TYPE vy = a->y;
  TYPE vz = a->z;
  
  const TYPE scal = (vx*b.x + vy*b.y + vz*b.z)/(b.x*b.x +
						b.y*b.y+
						b.z*b.z);
  
  vx = 2.0*scal*b.x - vx;
  vy = 2.0*scal*b.y - vy; 
  vz = 2.0*scal*b.z - vz;
  
  TYPE normN = RSQRT(vx*vx + vy*vy + vz*vz);
  a->x = vx * normN;
  a->y = vy * normN;
  a->z = vz * normN;
}

__device__ __host__ void _CudaGenerateVectorWithinSolidAngleFromAxeDevice(const TYPE CosThetaSolid, 
									  const TYPE3 in, 
									  TYPE Rng1,
									  TYPE Rng2,
									  TYPE3 *__restrict__ out)
{
  TYPE ts, tc, ps, pc;
  //SINCOS(theta, &ts, &tc);
  tc = ONE-(ONE-CosThetaSolid)*Rng1;
  ts = SQRT(ONE-tc*tc);
  const bool test = (in.z>-0.99995);
  TYPE N0 = (in.x+!test);
  TYPE N1 = in.y;  
  TYPE N2 = (in.z+test);
  SINCOSPI(TWO*Rng2, &ps, &pc);
  TYPE r1 = ((test)?(ts*ps):(-tc));
  TYPE r2 = ((test)?(tc):(ts*ps));
  TYPE r0=ts*pc;

  // dot product
  // 

  // this->RS[0] = 2*N[0]*(N[0]*rt[0]+N[1]*rt[1]+N[2]*rt[2])-rt[0];
  //             = 2*N[0]*(N . rt) -rt[0];
  // this->RS    = 2*(N . rt) N - rt;
  TYPE scal = ONE/(ONE + ((test)?in.z:in.x));
  scal *= N0*r0+N1*r1+N2*r2;
  
  TYPE o0 = N0*scal-r0;
  TYPE o1 = N1*scal-r1;
  TYPE o2 = N2*scal-r2;
  
  TYPE norm = RSQRT(o0*o0 + o1*o1 + o2*o2);
  
  out->x=o0*norm;
  out->y=o1*norm;
  out->z=o2*norm;
}

__global__ void _CudaGenerateVectorWithinSolidAngleFromAxe(const TYPE CosThetaSolid, 
							   const TYPE *__restrict__ in, 
							   const TYPE *__restrict__ Rng, 
							   TYPE *__restrict__ out)
{
  int Id = blockIdx.x*blockDim.x+threadIdx.x;
  _CudaGenerateVectorWithinSolidAngleFromAxeDevice(CosThetaSolid, 
						   ((TYPE3 *)in)[Id], 
						   Rng[2*Id],
						   Rng[2*Id+1],
						   ((TYPE3 *)out)+Id);
}

void Magnetization(const TYPE *s, const int nspc, const dim3 n, double *m)
{
  m[0] = 0;
  m[1] = 0;
  m[2] = 0;
  for(auto i=0;i<n.x*n.y*nspc*n.z;i++) {
    m[0] += s[3*i];
    m[1] += s[3*i+1];
    m[2] += s[3*i+2];
  }
}


void TotalEnergyv2(std::vector<dev> &device, std::vector<double> &Energy)
{  
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);

    // Need to indicate in which stream are the calculations
    // it is IMPORTANT FOR the reduction
    
    // Calculating the total energy is always done in streamMc[1]
    // because it is linked to the extraction of the montecarlo datas.

    device[d].CurrentStream = device[d].streamMc[1];
    TotalEnergyGPU<<<device[d].Blocks, 
      BlockSpin, 
      device[d].SharedMemory, 
      device[d].CurrentStream>>>(device[d].DevSpins, device[d].Red);
  }
  Reduce<TYPE>(1, Energy.size(), device, &Energy[0]);
  
  for(int T=0;T<Energy.size();T++)
    Energy[T] *= 0.5; 
} 



#if   !defined(DISCRETE)
GLOBAL void _CudaQuadrupolarOrder(const TYPE *spin, TYPE *Quadrupole)
{
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
#ifdef _2D_
  int3 n;
  int2 thd;
  InitializeConstants2D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  int nspc = n.z;
#else
  int4 n;
  int3 thd;
  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  int nspc = n.w;
#endif

  const TYPE *Sx = spin + 3*Offset;
  const TYPE *Sy = Sx + nspc*ncell;
  const TYPE *Sz = Sy + nspc*ncell;
  
  __syncthreads();
  TYPE q[6]= {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  for(auto s=0;s<nspc;s++) {
    TYPE3 si;
    si.x = Sx[Cell+s*ncell];
    si.y = Sy[Cell+s*ncell];
    si.z = Sz[Cell+s*ncell];
    q[0] += si.x * si.x - 0.333333333333333333333;
    q[1] += si.x * si.y;
    q[2] += si.x * si.z;
    q[3] += si.y * si.y - 0.333333333333333333333;
    q[4] += si.y * si.z;
    q[5] += si.z * si.z- 0.333333333333333333333;
  }
  
  for(int s=0;s<6;s++)
    Quadrupole[6*TemperatureNumber*ncell + s*ncell + Cell] = q[s];
}

void CalculateQuadrupole(const int NumberOfTemperatures, 
			 std::vector<dev> &device, 
			 TYPE *OrderParameters)
{
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    device[d].CurrentStream = device[d].streamMc[3];
    _CudaQuadrupolarOrder<<<device[d].Blocks,BlockSpin,0,device[d].CurrentStream>>>(device[d].DevSpins, 
										    device[d].Red);
  }
  
  Reduce<TYPE>(6, NumberOfTemperatures, device, OrderParameters);
}

GLOBAL void CudaCalculateMonopoleSquare(const TYPE *spin, TYPE *Quadrupole)
{

  /* mm1= 0.5773502691896258*s0.x + 0.5773502691896258*s0.y + 0.5773502691896258*s0.z -  */
  /*   0.5773502691896258*s1.x - 0.5773502691896258*s1.y + 0.5773502691896258*s1.z +  */
  /*   0.5773502691896258*s2.x - 0.5773502691896258*s2.y - 0.5773502691896258*s2.z -  */
  /*   0.5773502691896258*s3.x + 0.5773502691896258*s3.y - 0.5773502691896258*s3.z; */

  const  double sz[4][3] = {{1, 1, 1}, {-1, -1, 1}, {1, -1, -1}, {-1, 1, -1}};
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
#ifdef _2D_
  int3 n;
  int2 thd;
  InitializeConstants2D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
#else
  int4 n;
  int3 thd;
  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
#endif
  
  double Charge[4] = {0.0, 0.0, 0.0, 0.0};
  const TYPE *Sx = spin + 3*Offset;
  const TYPE *Sy = Sx + 16*ncell;
  const TYPE *Sz = Sy + 16*ncell;
  
  for(int te=0;te<4;te++) {
    for(int si=0;si<4;si++) {
      const double scal = Sx[Cell+(4*te+si)*ncell]*sz[si][0] + 
	Sy[Cell+(4*te+si)*ncell]*sz[si][1] + 
	Sz[Cell+(4*te+si)*ncell]*sz[si][2]; 
      Charge[te]+=scal;
    }
    Charge[te] *= Charge[te];
    Charge[te] = fabs(sqrt(Charge[te]));
  }
  
  /* // site 3 */
  /* int *link = &__links[6*3]; */
  /* TYPE charge = (Spin[Cell+3*ncell].x*sz[3][0] +  */
  /* 		 Spin[Cell+3*ncell].y*sz[3][1] +  */
  /* 		 Spin[Cell+3*ncell].z*sz[3][2]); */
  /* for(int s=3;s<6;s++)  */
  /*   { */
  /*      int xi = ((signed char *)(&link[s]))[0]; */
  /*      int yi = ((signed char *)(&link[s]))[1]; */
  /*      int zi = ((signed char *)(&link[s]))[2]; */
  /*      int ni = ((signed char *)(&link[s]))[3]; */
       
  /*      int x1 = (thd.x + xi + n.x)%n.x; */
  /*      int y1 = (thd.y + yi + n.y)%n.y; */
  /*      int z1 = (thd.z + zi + n.z)%n.z; */
  /*      int cl = (z1*n.y+y1)*n.x+x1; */
  /*      charge =  (Spin[cl+ni*ncell].x*sz[ni%4][0] +  */
  /* 		  Spin[cl+ni*ncell].y*sz[ni%4][1] +  */
  /* 		  Spin[cl+ni*ncell].z*sz[ni%4][2]);   */
  /*   } */
  /* Charge[0] += fabs(charge); */
  
  /* //site 7 */
  /* link = &__links[6*7]; */
  /* charge =  (Spin[Cell+7*ncell].x*sz[3][0] +  */
  /* 	     Spin[Cell+7*ncell].y*sz[3][1] +  */
  /* 	     Spin[Cell+7*ncell].z*sz[3][2]); */
  /* for(int s=3;s<6;s++)  */
  /*   { */
  /*     int xi = ((signed char *)(&link[s]))[0]; */
  /*     int yi = ((signed char *)(&link[s]))[1]; */
  /*     int zi = ((signed char *)(&link[s]))[2]; */
  /*     int ni = ((signed char *)(&link[s]))[3]; */
      
  /*     int x1 = (thd.x + xi + n.x)%n.x; */
  /*     int y1 = (thd.y + yi + n.y)%n.y; */
  /*     int z1 = (thd.z + zi + n.z)%n.z; */
  /*     int cl = (z1*n.y+y1)*n.x+x1 + ni * ncell; */
  /*     charge +=  (Spin[cl].x*sz[ni%4][0] +  */
  /* 		  Spin[cl].y*sz[ni%4][1] +  */
  /* 		  Spin[cl].z*sz[ni%4][2]);   */
  /*   } */

  /* Charge[1] += fabs(charge); */

  /* //site 11 */
  /* link = &__links[6*11]; */
  /* charge = (Spin[Cell+11*ncell].x*sz[3][0] +  */
  /* 	    Spin[Cell+11*ncell].y*sz[3][1] +  */
  /* 	    Spin[Cell+11*ncell].z*sz[3][2]); */
  /* for(int s=3;s<6;s++)  */
  /*   { */
  /*     int xi = ((signed char *)(&link[s]))[0]; */
  /*     int yi = ((signed char *)(&link[s]))[1]; */
  /*     int zi = ((signed char *)(&link[s]))[2]; */
  /*     int ni = ((signed char *)(&link[s]))[3]; */

  /*     int x1 = (thd.x + xi + n.x)%n.x; */
  /*     int y1 = (thd.y + yi + n.y)%n.y; */
  /*     int z1 = (thd.z + zi + n.z)%n.z; */
  /*     int cl = (z1*n.y+y1)*n.x+x1 + ni * ncell; */
  /*     charge +=  (Spin[cl].x*sz[ni%4][0] + Spin[cl].y*sz[ni%4][1] + Spin[cl].z*sz[ni%4][2]);   */
  /*   } */
  /* Charge[2] += fabs(charge); */
  
  /* //site 15 */
  /* charge = (Spin[Cell+15*ncell].x*sz[3][0] +  */
  /* 	    Spin[Cell+15*ncell].y*sz[3][1] +  */
  /* 	    Spin[Cell+15*ncell].z*sz[3][2]); */
  /* link = &__links[6*15]; */
  /* for(int s=3;s<6;s++)  */
  /*   { */
  /*     int xi = ((signed char *)(&link[s]))[0]; */
  /*     int yi = ((signed char *)(&link[s]))[1]; */
  /*     int zi = ((signed char *)(&link[s]))[2]; */
  /*     int ni = ((signed char *)(&link[s]))[3]; */

  /*     int x1 = (thd.x + xi + n.x)%n.x; */
  /*     int y1 = (thd.y + yi + n.y)%n.y; */
  /*     int z1 = (thd.z + zi + n.z)%n.z; */
  /*     int cl = (z1*n.y+y1)*n.x+x1 + ni * ncell; */
  /*     charge +=  (Spin[cl].x*sz[ni%4][0] +  */
  /* 		  Spin[cl].y*sz[ni%4][1] +  */
  /* 		  Spin[cl].z*sz[ni%4][2]);   */
  /*   } */

  /* Charge[3] += fabs(charge);  */
  Quadrupole[TemperatureNumber*ncell + Cell] = (Charge[0] + Charge[1] + Charge[2] + Charge[3])/4.0; // divice by 2 because one half spin and 4 because
  //we have 4 tetrahedra per unit cell of type A. The total number of monopole is twice this 
}

GLOBAL void CudaCalculateAbsSz(const TYPE *spin, TYPE *Quadrupole)
{
  const  double s[4][3] = {{1, 1, 1}, {-1, -1, 1}, {1, -1, -1}, {-1, 1, -1}};
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
#ifdef _2D_
  int3 n;
  int2 thd;
  InitializeConstants2D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  int nspc = n.z;
#else
  int4 n;
  int3 thd;
  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  int nspc = n.w;
#endif
  
  double A = 0.0;
  const TYPE *Sx = spin + 3*Offset;
  const TYPE *Sy = Sx + nspc*ncell;
  const TYPE *Sz = Sy + nspc*ncell;
  
  for(int te=0;te<4;te++) {
    for(int si=0;si<4;si++) {
      const double scal = Sx[Cell+(4*te+si)*ncell]*s[si][0] + 
	Sy[Cell+(4*te+si)*ncell]*s[si][1] + 
	Sz[Cell+(4*te+si)*ncell]*s[si][2]; 
      A+=fabs(scal);
    }
  }
  
  Quadrupole[TemperatureNumber*ncell + Cell] = A/sqrt(3.0);
}
#endif

GLOBAL void CudaCalculateIsingConf(const TYPE *spin, int *Ising)
{
  const  double s[4][3] = {{1, 1, 1}, {-1, -1, 1}, {1, -1, -1}, {-1, 1, -1}};
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
#ifdef _2D_
  int3 n;
  int2 thd;
  InitializeConstants2D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  int nspc = n.z;
#else
  int4 n;
  int3 thd;
  InitializeConstants3D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  int nspc = n.w;
#endif
  
  const TYPE *Sx = spin + 3*Offset;
  const TYPE *Sy = Sx + nspc*ncell;
  const TYPE *Sz = Sy + nspc*ncell;
  int cs=0;
  for(int te=0;te<4;te++) {
    for(int si=0;si<4;si++) {
      const double scal = Sx[Cell+(4*te+si)*ncell]*s[si][0] + 
	Sy[Cell+(4*te+si)*ncell]*s[si][1] + 
	Sz[Cell+(4*te+si)*ncell]*s[si][2];
      cs = cs ^ (scal<0)<<(4*te+si);
    }
  }

  Ising[Cell] = cs;
}


static void _AoSToSoA(const void *a, void *b, const int elem, const int deg, const int size)
{
  // s1x,s1y,s1z,s2x,s2y,s2y

  // s1x,s2x,....
  if(size == sizeof(double)) {
    const double *ad = (const double *)a;
    double *bd = (double *)b;
    for(int d=0;d<deg;d++)
      vdPackI(elem, ad+d, deg, bd+d*elem);
  }
  if(size == sizeof(float)) {
    const float *ad = (const float *)a;
    float *bd = (float *)b;
    for(int d=0;d<deg;d++)
      vsPackI(elem, ad+d, deg, bd+d*elem);
  }
}

 static void _SoAToAoS(const void *a, void *b, const int elem, const int deg, const int size)
{
  if(size == sizeof(double)) {
    const double *ad = (const double *)a;
    double *bd = (double *)b;
    for(int d=0;d<deg;d++)
      vdUnpackI(elem, ad+d*elem, bd+d, deg);
  }
  if(size == sizeof(float)) {
    const float *ad = (const float *)a;
    float *bd = (float *)b;
    for(int d=0;d<deg;d++)
      vsUnpackI(elem, ad+d*elem, bd+d, deg);
  }
}


template <typename T> void SaveConfigurations(const T **Confs, 
					      const vector<double> &Temperatures, 
					      const int elem, 
					      const int deg,
					      const string filename, 
					      const int append)
{
  std::vector <size_t>  ConfigurationsPermutation(Temperatures.size());
  gsl_sort_largest_index (&ConfigurationsPermutation[0], 
			  Temperatures.size(), 
			  &Temperatures[0], 
			  1, 
			  Temperatures.size());
  char *name = (char *)malloc(sizeof(char)*256);
  std::vector <T> Spins(elem*deg);
  for(auto T1=0;T1<Temperatures.size();T1++) {
    memset(name, 0, 256);
    sprintf(name,"%s%d.dat", filename.c_str(), T1);
    FILE *f = NULL;
    if(append)
      f = fopen(name,"a+");
    else
      f = fopen(name,"w+");

    if(typeid(T)==typeid(double)) {     
      _SoAToAoS((const void *)&Confs[ConfigurationsPermutation[T1]][0], (void *)&Spins[0], elem, deg, sizeof(double));
      fwrite(&Spins[0], sizeof(double), elem*deg, f);
    }
    if(typeid(T)==typeid(float)) {
      _SoAToAoS((const void *)&Confs[ConfigurationsPermutation[T1]][0], (void *)&Spins[0], elem, deg, sizeof(float));    
      fwrite(&Spins[0], sizeof(float), elem*deg, f);
    }
    if(typeid(T) == typeid(int))
      fwrite(&Confs[ConfigurationsPermutation[T1]][0], sizeof(int), elem*deg, f);
    
    fclose(f);
  }
  free(name);
  Spins.clear();
  ConfigurationsPermutation.clear();
}



template <typename T> int ReadConfigurations(T **Confs, 
					     const int NumberOfTemperatures, 
					     const int elem, 
					     const int deg,
					     const string filename, 
					     const int rep)
{
  char *name = (char *)malloc(sizeof(char)*256);
  std::vector <T> Spins(elem*deg);
  for(int T1=0;T1<NumberOfTemperatures;T1++) {
    memset(name,0,256);
    sprintf(name,"%s%d.dat", filename.c_str(), T1);
    //printf("%s\n", name);
    FILE *f = fopen(name,"r");
    if(!f) {
      printf("Error : the equilibration is not done or you are in the wrong directory\n");
      printf(" Please check that the directory contains files named InitialConf*.dat");
      exit(1);
    }

    fseek(f, rep*elem*deg*sizeof(T), SEEK_SET);
    if(fread(&Spins[0], sizeof(T), elem*deg, f) != (elem*deg))
      {
	fclose(f);
	free(name);
	printf("Error impossible to read the file. We arrived at the end of it\n");
	return 1;
      }
    fclose(f);
    if(typeid(T)==typeid(double))
      _AoSToSoA((const void *)&Spins[0], (void *)&Confs[T1][0], elem, deg, sizeof(double));
    if(typeid(T)==typeid(float))
      _AoSToSoA((const void *)&Spins[0], (void *)&Confs[T1][0], elem, deg, sizeof(float));    
    if(typeid(T) == typeid(int))
      memcpy(&Confs[T1][0], &Spins[0], sizeof(int)*elem*deg);
  }
  free(name);
  Spins.clear();
  return 0;
}

template void SaveConfigurations<int>(const int **Confs, const std::vector<double> &Temperatures, const int elem, const int deg, const string name, const int Append);
template void SaveConfigurations<double>(const double **Confs, const std::vector<double> &Temperatures, const int elem, const int deg, const string name, const int Append);
template void SaveConfigurations<float>(const float **Confs, const std::vector<double> &Temperatures, const int elem, const int deg,  const string name, const int Append);
template int ReadConfigurations<int>(int **Confs, const int NumberOfTemperatures, const int elem, const int deg,  const string name, const int rep);
template int ReadConfigurations<double>(double **Confs, const int NumberOfTemperatures, const int elem, const int deg,  const string name, const int rep);
template int ReadConfigurations<float>(float **Confs, const int NumberOfTemperatures, const int elem, const int deg,  const string name, const int rep);

__global__  void _vzabsgpu(const int length, const TYPE *a, TYPE *c)
{
  if(threadIdx.x+blockIdx.x*blockDim.x>=length)
    return;
  
  TYPE2 aa;
  aa = ((TYPE2 *)a)[blockIdx.x*blockDim.x + threadIdx.x];
    
  c[threadIdx.x+blockIdx.x*blockDim.x] = aa.x*aa.x + aa.y*aa.y;
}

__global__  void _vzabsmtgpu(const int length, const int deg, const TYPE *a, TYPE *c)
{
  if(threadIdx.x+blockIdx.x*blockDim.x>=length)
    return;
  
  TYPE2 *shared_a = (TYPE2 *)shared_mem;
  const TYPE2 *aa = (TYPE2 *)a;
  for(int d=0;d<deg;d++) {
    shared_a[deg*threadIdx.x+d] = aa[deg*(blockIdx.x*blockDim.x + threadIdx.x) + d];
  }
  __syncthreads();
  
  TYPE sum = shared_a[deg*threadIdx.x].x*shared_a[deg*threadIdx.x].x + 
    shared_a[deg*threadIdx.x].y*shared_a[deg*threadIdx.x+1].y;
  
  for(int d=1;d<deg;d++)
    sum += shared_a[deg*threadIdx.x+d].x*shared_a[deg*threadIdx.x+d].x + 
      shared_a[deg*threadIdx.x+d].y*shared_a[deg*threadIdx.x+d].y;
  
  c[threadIdx.x+blockIdx.x*blockDim.x] += sum;
}

__global__ void _ComputeProjectedSqGPU(const int length, 
				       const int deg, 
				       const int _SpinFlipNonSpinFlip, 
				       const TYPE px, 
				       const TYPE py, 
				       const TYPE pz, 
				       const TYPE *a, 
				       const TYPE *qp, 
				       TYPE *c)
{
  if((threadIdx.x+blockIdx.x*blockDim.x)>=length)
    return;
  
  TYPE2 *shared_a = (TYPE2 *)shared_mem;
  TYPE q[3];
  q[0] = qp[3*(threadIdx.x+blockIdx.x*blockDim.x)];
  q[1] = qp[3*(threadIdx.x+blockIdx.x*blockDim.x)+1];
  q[2] = qp[3*(threadIdx.x+blockIdx.x*blockDim.x)+2];

  const TYPE2 *aa = (TYPE2*)a;
  
  
  for(int d=0;d<deg;d++) {
    shared_a[deg*threadIdx.x+d] = aa[deg*(blockIdx.x*blockDim.x + threadIdx.x) + d ];
  }

  TYPE norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
  if((threadIdx.x+blockIdx.x*blockDim.x) == 0) 
    norm = 0;
  else
    norm = 1.0/norm;
  

  __syncthreads();
  
  TYPE3 res;
  res.x = 0.0;
  for(int d1=0;d1<deg;d1++) {
    TYPE2 mtq1;
    mtq1 = shared_a[deg*threadIdx.x+d1];
    for(int d2=0;d2<deg;d2++) 
      {
	TYPE2 mtq2;
	mtq2 = shared_a[deg*threadIdx.x+d2];
	res.x += ((d2==d1)*1.0 - q[d1]*q[d2]*norm) *(mtq1.x*mtq2.x + mtq1.y*mtq2.y);
      }
  }
  
  if(_SpinFlipNonSpinFlip)
    {
      TYPE2 scal;
      // NSF component
      scal.x = shared_a[deg*threadIdx.x].x * px + 
	shared_a[deg*threadIdx.x+1].x * py + 
	shared_a[deg*threadIdx.x+2].x * pz; //Re(ab)
      scal.y = shared_a[deg*threadIdx.x].y * px + 
	shared_a[deg*threadIdx.x+1].y * py + 
	shared_a[deg*threadIdx.x+2].y * pz; // Im(ab)
      res.y = scal.x*scal.x+scal.y*scal.y;
      
      //SF component
      
      TYPE3 pXq;
      
      pXq.x = py*q[2] - pz*q[1];
      pXq.y = pz*q[0] - px*q[2];
      pXq.z = px*q[1] - py*q[0];
      // m_q . (n x q)
      scal.x = shared_a[deg*threadIdx.x].x * pXq.x + shared_a[deg*threadIdx.x+1].x * pXq.y + shared_a[deg*threadIdx.x+2].x * pXq.z; //Re(ab)
      scal.y = shared_a[deg*threadIdx.x].y * pXq.x + shared_a[deg*threadIdx.x+1].y * pXq.y + shared_a[deg*threadIdx.x+2].y * pXq.z; // Im(ab)
      // |m_q x (n x q)|^2
      res.z = (scal.x*scal.x+scal.y*scal.y)*norm;
      
    }
  
  c[threadIdx.x+blockIdx.x*blockDim.x] += res.x;
  c[threadIdx.x+blockIdx.x*blockDim.x + length] += res.y;
  c[threadIdx.x+blockIdx.x*blockDim.x + 2*length] += res.z;
}

__global__ void _Compute_SF_NSF_GPU(const int length,
                                    const int deg,
                                    const TYPE *__restrict__ a,
                                    const TYPE *__restrict__ qp,
                                    const TYPE px,
                                    const TYPE py,
                                    const TYPE pz,
                                    TYPE *__restrict__ c)
{
  if((threadIdx.x+blockIdx.x*blockDim.x)>=length)
    return;
  
  TYPE2 *shared_a = (TYPE2 *)shared_mem;
  TYPE q[3];
  q[0] = qp[3*(threadIdx.x+blockIdx.x*blockDim.x)];
  q[1] = qp[3*(threadIdx.x+blockIdx.x*blockDim.x)+1];
  q[2] = qp[3*(threadIdx.x+blockIdx.x*blockDim.x)+2];

  TYPE norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
  const TYPE2 *aa = (TYPE2*)a;
  
  if((threadIdx.x+blockIdx.x*blockDim.x) == 0) 
    norm = 0;
  else
    norm = 1.0/norm;
  
  for(int d=0;d<deg;d++) {
    shared_a[deg*threadIdx.x+d] = aa[deg*(blockIdx.x*blockDim.x + threadIdx.x) + d ];
  }
  __syncthreads();
  
  TYPE2 scal;
  TYPE2 res;

  // NSF component
  scal.x = shared_a[deg*threadIdx.x].x * px + shared_a[deg*threadIdx.x+1].x * py + shared_a[deg*threadIdx.x+2].x * pz; //Re(ab)
  scal.y = shared_a[deg*threadIdx.x].y * px + shared_a[deg*threadIdx.x+1].y * py + shared_a[deg*threadIdx.x+2].y * pz; // Im(ab)
  res.x = scal.x*scal.x+scal.y*scal.y;

  //SF component

  TYPE3 pXq;
  
  pXq.x = py*q[2] - pz*q[1];
  pXq.y = pz*q[0] - px*q[2];
  pXq.z = px*q[1] - py*q[0];

  scal.x = shared_a[deg*threadIdx.x].x * pXq.x + shared_a[deg*threadIdx.x+1].x * pXq.y + shared_a[deg*threadIdx.x+2].x * pXq.z; //Re(ab)
  scal.y = shared_a[deg*threadIdx.x].y * pXq.x + shared_a[deg*threadIdx.x+1].y * pXq.y + shared_a[deg*threadIdx.x+2].y * pXq.z; // Im(ab)
  
  res.y = scal.x*scal.x+scal.y*scal.y;

  c[threadIdx.x+blockIdx.x*blockDim.x] += res.x;
  c[threadIdx.x+blockIdx.x*blockDim.x + length] += res.y*norm;
}


void vzMulGPU( cudaStream_t stream, const int length, const TYPE *a, const TYPE *b, TYPE *c)
{
  dim3 block,thread;
  block.x = length/256 + (length%256);
  thread.x = 256;
  _vzmulgpu<<<block.x, thread.x, 0, stream>>>(length, a, b, c);
}

void vzAbs2GPU( cudaStream_t stream, const int length, const TYPE *a, TYPE *c)
{
  int sharemem = 2*256;
  dim3 block,thread;
  block.x = length/256 + (length%256);
  thread.x = 256;
  _vzabsgpu<<<block.x, thread.x, sharemem*sizeof(TYPE), stream>>>(length, a, c);
}

void ComputeNormMagnetizationGPU(cudaStream_t stream, const int length, const int deg, const TYPE *a, TYPE *c)
{
  int sharemem = 0;
  dim3 block,thread;
  sharemem = 2*deg*256;
  block.x = length/256 + (length%256!=0);
  thread.x = 256;
  if(deg != 1)
    _vzabsmtgpu<<<block, thread, sharemem*sizeof(TYPE), stream>>>(length, deg, a, c);
  else
    _vzabsgpu<<<block.x, thread.x, sharemem*sizeof(TYPE), stream>>>(length, a, c);
}


void ComputeProjectedSqGPU(cudaStream_t stream, const int length, const int deg, const int _Projected, const TYPE *p, const TYPE *mt, const TYPE *qp, TYPE *c)
{
  int sharemem = 0;
  dim3 block,thread;
  sharemem = deg*256;
  block.x = length/256 + (length%256!=0);
  thread.x = 256;
  _ComputeProjectedSqGPU<<<block, thread, sharemem*sizeof(TYPE2), stream>>>(length, deg, _Projected, p[0], p[1], p[2], mt, qp, c);
}

void ComputeSF_NSF_GPU(cudaStream_t stream, const int length, const int deg, const TYPE *mt, const TYPE *qp, const TYPE *p, TYPE *c)
{
  int sharemem = 0;
  dim3 block,thread;
  sharemem = deg*256;
  block.x = length/256 + (length%256!=0);
  thread.x = 256;
  _Compute_SF_NSF_GPU<<<block, thread, sharemem*sizeof(TYPE2), stream>>>(length, deg, mt, qp, p[0], p[1], p[2], c);
}

GLOBAL void PrepareDataForFFTPotts(const int *data, TYPE *d)
{
  int3 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int2 thd;
  InitializeConstants2D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  const int mask[6] = {0x3,0xC,0x30,0xc0,0x300,0xc00};
  const int cf = data[Cell+TemperatureNumber*ncell];
  double *Spins = d + Offset*12;

  for(int s=0;s<6;s++) {
    Spins[2*(Cell + s*ncell)] = (cf & mask[s])>>(2*s)-1;
    Spins[2*(Cell + s*ncell)+1] = 0;
  }
}

GLOBAL void PrepareDataForFFTClockModel(const int *data, TYPE *d)
{
  const int mask[6] = {0x3,0xC,0x30,0xc0,0x300,0xc00};
  int3 n;
  int TemperatureNumber, blk, tht, LocalCell, Cell, ncell, Offset;
  int2 thd;
  InitializeConstants2D(&n, &blk, &TemperatureNumber, &thd, &tht, &LocalCell, &Cell, &ncell, &Offset);
  const int cf = data[Cell+TemperatureNumber*ncell];
  const double color[3][2] = {{1.0,0.0},{-0.5,sqrt(3.0)*0.5},{-0.5,-sqrt(3.0)*0.5}};

  double *Sx = d + TemperatureNumber*ncell*24; // 6 spins 2 coordinates per spin times 2 because complex numbers
  double *Sy = Sx + ncell*12;

  for(int s=0;s<6;s++) {
    Sx[2*(Cell+s*ncell)] = color[(cf & mask[s])>>(2*s)][0];
    Sx[2*(Cell+s*ncell)+1] = 0.0;
    Sy[2*(Cell+s*ncell)] = color[(cf & mask[s])>>(2*s)][1];
    Sy[2*(Cell+s*ncell)+1] = 0.0;
  }
}

inline __device__ int Metropolis(const TYPE3 ExchangeField, const TYPE InverseTemperature, const TYPE3 *__restrict__ tmp, const TYPE rng, TYPE3 *__restrict__ out)
{
  /* Energy = - sum_j  (J sum_<i,j> s_i) s_j */
  const TYPE LocalEnergyBefore = -(ExchangeField.x * tmp->x + ExchangeField.y * tmp->y + ExchangeField.z * tmp->z);
  /* // call the RandomSpin routine */
  const TYPE LocalEnergyAfter = -(ExchangeField.x * out->x + ExchangeField.y * out->y + ExchangeField.z * out->z);
  const TYPE DE = LocalEnergyBefore - LocalEnergyAfter;
  // the factor 0.5 is for the spin one-half
  
  const TYPE BoltzmanWeight = EXP(DE*InverseTemperature);
  const int test = rng < min(1.0, BoltzmanWeight);

  
  out->x = out->x*test + tmp->x*(!test);
  out->y = out->y*test + tmp->y*(!test);
  out->z = out->z*test + tmp->z*(!test);

  return test;
}

inline __device__ int CalculateNeighbor3D(const int *link, const int3 thd, const int4 n, int *__restrict__ localIndex)
{
  int xi = ((signed char *)(link))[0];
  int yi = ((signed char *)(link))[1];
  int zi = ((signed char *)(link))[2];
  int ni = ((signed char *)(link))[3];

  int x1 = thd.x + xi + n.x;    
  while(x1>= n.x)
    x1-= n.x;
  int y1 = thd.y + xi + n.y;
  while(y1>= n.y)
    y1-= n.y;
  int z1 = thd.z + yi + n.z;
  while(z1>= n.z)
    z1-= n.z;
  
  if(localIndex)
    localIndex[0] = (((int)threadIdx.z + zi)*n.y + (int)threadIdx.y + yi)*n.x + (int)threadIdx.x + xi;

  return ((z1*n.y+y1)*n.x+x1) + ni*n.x*n.y*n.z;
}

inline __device__ int CalculateNeighbor2D(const int *link, const int2 thd, const int3 n, int *__restrict__ localIndex)
{
  int xi = ((signed char *)(link))[0];
  int yi = ((signed char *)(link))[1];
  int ni = ((signed char *)(link))[2];

  int x1 = thd.x + xi + n.x;    
  while(x1>= n.x)
    x1-= n.x;
  int y1 = thd.y + xi + n.y;
  while(y1>= n.y)
    y1-= n.y;
  
  if(localIndex)
    localIndex[0] = ((int)threadIdx.y + yi)*n.x + (int)threadIdx.x + xi;

  return (y1*n.x+x1) + ni*n.x*n.y;
}

#include "MonteCarlo.cu"
#include "Device.cu"
