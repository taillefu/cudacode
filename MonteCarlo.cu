static int _Sync = 0;

void GenerateNewRandomNumbers(dev *Device, const int DeviceId, const unsigned int n, const int NumberOfTemperatures)
{
  if(Device[DeviceId].NumberOfTemperatures < NumberOfTemperatures) {
    printf("Bad parameter value for the number of temperatures. It should lower than the number of temperatures on one device");
  }
#ifdef USE_DOUBLE
  curandGenerateUniformDouble(Device[DeviceId].Gen, Device[DeviceId].RngDev, n*NumberOfTemperatures);
#else
  curandGenerateUniform(Device[DeviceId].Gen, Device[DeviceId].RngDev, n*NumberOfTemperatures);
#endif
}

void *MonteCarloThread(void *ThreadData)
{
  
  // Generate a list of parallel updates uniformaly.  Since we work
  // on lattices the number of simulataneous sublattice updates is
  // finite but depends on the lattice and unit cell.
  
  // For the pyrochlore lattice with extended unit cell we have 24
  // possible sublattices updates while for the cubic lattice only
  // two
  
  // each update updates NumberOfSubLattices in one sweep and we
  // have NumberOfSpinsPerUnitCell sublattices so the number of loop should
  // be. NumberOfSpinsPerUnitCell/NumberOfSubLattices
  
  // I only pick RatioUniCellSubLattices  amongst all possible updates.
  
  
  dev *device = (dev *)ThreadData;
    
  cudaSetDevice(device->DeviceId);
#ifdef ENABLE_TIMING
  cudaEvent_t startEvent, stopEvent;
  if(device[0].DeviceId == 0) {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent,0);
  }
#endif

  cudaMemset(device[0].DevResults, 0, sizeof(int)*device[0].ncell*device[0].NumberOfTemperatures);
  
  Mc(device);

#ifdef ENABLE_TIMING
  if(device[0].DeviceId == 0) {
    cudaEventRecord(stopEvent,0);
    cudaEventSynchronize(stopEvent);
    
    float millisecond;
    cudaEventElapsedTime(&millisecond, startEvent, stopEvent);
    
    printf("%.10lf ms for %d Mcsteps on a %d nspins system\n", millisecond, device[0].steps, device[0].nspins);
    printf("%.10lf ms per spin\n", ((double)millisecond)/(((double)device[0].steps)*(((double)device[0].nspins))));
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
  }
#endif
  pthread_exit(NULL);
}


void *SimulatedAnnealingThread(void *ThreadData)
{
  
  // Generate a list of parallel updates uniformaly.  Since we work
  // on lattices the number of simulataneous sublattice updates is
  // finite but depends on the lattice and unit cell.
  // For the pyrochlore lattice with extended unit cell we have 24
  // possible sublattices updates while for the cubic lattice only
  // two
  
  // each update updates NumberOfSubLattices in one sweep and we
  // have NumberOfSpinsPerUnitCell sublattices so the number of loop should
  // be. NumberOfSpinsPerUnitCell/NumberOfSubLattices
  
  // I only pick RatioUniCellSubLattices  amongst all possible updates.

  dev *device = (dev *)ThreadData;
  double *Temp = (double *)malloc(sizeof(double)*device->NumberOfTemperatures);
  
  cudaSetDevice(device->DeviceId);

  for(auto m=0;m<device->SimulatedAnnealingSteps;m++) {
    for(auto T=0;T<device->NumberOfTemperatures;T++) {
      int TempIndex = device->DeviceId*device->NumberOfTemperatures+T;
      Temp[T] = device->StartingTemperature - (device->StartingTemperature-device->TemperaturesTarget[TempIndex])*m*device->DeltaT;
    }
    cudaMemcpyAsync(device[0].Temperatures,
    		    Temp,
    		    sizeof(double)*device[0].NumberOfTemperatures,
    		    cudaMemcpyHostToDevice,
    		    device[0].streamMcpy[0]);
    Mc(device);
  }

  free(Temp);
  pthread_exit(NULL);
}
void TrySwapingTemperatures(
#if !defined(DISCRETE)
			    vector<TYPE> &CosThetaSolid,
#endif			    
			    vector<double> &Energies,
			    vector<double> &Temperatures,
			    int EvenOdd)
{
  TYPE *rng = (TYPE *)malloc(sizeof(TYPE)*Temperatures.size());
  std::vector<size_t> ConfigurationsPermutation(Temperatures.size(),0);
#ifdef USE_DOUBLE
  vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, RngStream, Temperatures.size(), rng, 0, 1);
#else
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, RngStream, Temperatures.size(), rng, 0, 1);
#endif
  gsl_sort_largest_index (&ConfigurationsPermutation[0], Temperatures.size(), &Temperatures[0], 1, Temperatures.size());    
  
  if(EvenOdd) {
    for(auto c=0;c<Temperatures.size();c+=2) {
      int t1 = ConfigurationsPermutation[c];
      int t2 = ConfigurationsPermutation[c+1];
      
      double E1 = Energies[t1];
      double E2 = Energies[t2];
      double T1 = Temperatures[t1]; // b high
      double T2 = Temperatures[t2]; // b low
      double Weight = min(1.0, exp((1.0/T1 - 1.0/T2)*(E1 - E2)));
      
      if(rng[c] < Weight) {
	Temperatures[t1] = T2;
	Temperatures[t2] = T1;
#if !defined(DISCRETE)
	if(CosThetaSolid.size()) {	
	  TYPE swap = CosThetaSolid[t1];
	  CosThetaSolid[t1] = CosThetaSolid[t2];
	  CosThetaSolid[t2] = swap;
	}
#endif
      }
    }
  } else {
    for(auto c=1;c<Temperatures.size()-1;c+=2) {
      int t1 = ConfigurationsPermutation[c];
      int t2 = ConfigurationsPermutation[c+1];
      //printf("%d\n",tmp[c]);
      double E1 = Energies[t1]; 
      double E2 = Energies[t2];
      double T1 = Temperatures[t1];
      double T2 = Temperatures[t2];
      double Weight = min(1.0, exp((1.0/T1 - 1.0/T2)*(E1 - E2)));
      
      if(rng[c] < Weight) {
	/* printf("%.5lf\n",Weight); */
	Temperatures[t1] = T2;
	Temperatures[t2] = T1;
#if  !defined(DISCRETE)
	if(CosThetaSolid.size()) {
	  TYPE swap = CosThetaSolid[t1];
	  CosThetaSolid[t1] = CosThetaSolid[t2];
	  CosThetaSolid[t2] = swap;
	}
#endif
      }	
    }
  }
  ConfigurationsPermutation.clear();
  free(rng);
}



void MonteCarlo(std::vector<dev> &device,
#if   !defined(DISCRETE)
		const int OverRelaxation,
		const int ReductionAngle,
#endif
		const vector<int> &McSteps,
#if   !defined(DISCRETE)
		std::vector<double> &Temperatures,
		std::vector<TYPE> &CosThetaSolid,
		TYPE *ratio
#else
		std::vector<double> &Temperatures
#endif
		)
{

  // The number of temperatures on the devices is smaller than the
  // total number of temperatures. if not the confs are assigned to a
  // GPU. It is the temperatures that are shuffled on the GPUs

  _Sync = 0;

  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    device[d].steps = McSteps[0];
    
#if  !defined(DISCRETE)
    device[d].OverRelaxation = OverRelaxation;
    device[d].ReductionAngle = ReductionAngle;

    if((ReductionAngle)&&(CosThetaSolid.size())) {
      cudaMemcpyAsync(device[d].CosThetaSolid, 
		      &CosThetaSolid[d*device[d].NumberOfTemperatures],
		      sizeof(double)*device[d].NumberOfTemperatures, 
		      cudaMemcpyHostToDevice, device[d].streamMcpy[0]);
    }
#endif
  }

  pthread_attr_init(&device[0].attr);
  pthread_attr_setdetachstate(&device[0].attr, PTHREAD_CREATE_JOINABLE);  
  
  for(auto d=0;d<device.size();d++) {
    device[d].DeviceId = d;
    int    rc = pthread_create(&device[d].threads, NULL, MonteCarloThread, (void *)&device[d]);
    if (rc){
      printf("Error:unable to create thread, %d\n",rc);
      exit(-1);
    }
  }
  _Sync = 1;
  pthread_attr_destroy(&device[0].attr);
}

void SynchronizeJobs(vector<dev> &device)
{
  void *status;  

  if(_Sync) {
    for(auto d=0;d<device.size();d++) {
      int rc = pthread_join(device[d].threads, &status);
      if (rc){
	printf("Error:unable to join, %d",rc);
	exit(-1);
      }
    }
  }

  _Sync=0;
}

void CleanUp(vector<dev> &device)
{
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    cudaFree(device[d].Rng);
    device[d].RngSize = 0;
    device[d].Rng = NULL;
  }
}

void *RandomVectorOnDeviceThread(void *data)
{
  dev *device = (dev *)data;
  cudaSetDevice(device->DeviceId);
  GenerateNewRandomNumbers(device, 0, 2*device->nspins, device->NumberOfTemperatures);
#if   !defined(DISCRETE)
  Generate3DVectorList<<<device->nspins*device->NumberOfTemperatures/64, 64>>>(device->RngDev,
									       device->nspins*device->NumberOfTemperatures,
									       device->Spins);
#else
  GeneratePottsStates<<<device->ncell*device->NumberOfTemperatures/64, 64>>>(device->RngDev, 
									     device->ncell*device->NumberOfTemperatures,
									     device->Spins);
  
#endif
  pthread_exit(NULL);
}

void GenerateRandomVectorsOnDevice(std::vector<dev> &device)
{
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  
  for(auto d=0;d<device.size();d++) {
    device[d].DeviceId = d;
    int rc = pthread_create(&device[d].threads, NULL, RandomVectorOnDeviceThread, (void *)&device[d] );
    if (rc){
      printf("Error:unable to create thread, %d", rc);
      exit(-1);
    }
  }

  pthread_attr_destroy(&attr);
  _Sync = 1;
  SynchronizeJobs(device);
}

void SimulatedAnnealing(std::vector<dev> &device,
			const int GenerateVectors,
			const int SimulatedAnnealingSteps,
#if   !defined(DISCRETE)
			const int OverRelaxation,
			const int ReductionAngle,
#endif
			const std::vector<int> &McSteps,
			vector<double> &Temperatures,
#if !defined(DISCRETE)
			double StartingTemperature,
			vector<TYPE> &CosThetaSolid			
#else
			double StartingTemperature
#endif
			)

{

  const double DeltaT = 1/((double)SimulatedAnnealingSteps);
  pthread_attr_t attr;

  if(GenerateVectors)
    GenerateRandomVectorsOnDevice(device);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  _Sync = 1;
  for(auto d=0;d<device.size();d++) {
#if   !defined(DISCRETE)
    device[d].OverRelaxation = OverRelaxation;
    device[d].ReductionAngle = ReductionAngle;
#endif
    device[d].DeviceId = d;
    device[d].SimulatedAnnealingSteps = SimulatedAnnealingSteps;
    device[d].TemperaturesTarget = &Temperatures[0];
    device[d].StartingTemperature = StartingTemperature;
    device[d].DeltaT = DeltaT;
    device[d].steps = McSteps[0];

    int rc = pthread_create(&device[d].threads, NULL, SimulatedAnnealingThread, (void *)&device[d]);
    if (rc){
      printf("Error:unable to create thread, %d", rc);
      exit(-1);
    }
  }
  pthread_attr_destroy(&attr);
  _Sync = 1;
  SynchronizeJobs(device);
}

void CalculateRatio(const int NumberOfTemperatures, 
		    const int McSteps, 
		    std::vector<dev> &device, 
		    TYPE *Ratio)
{
  if(!Ratio)
    return;
  memset(Ratio, 0, sizeof(TYPE)*NumberOfTemperatures);
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    cudaMemcpy(device[d].RngDev,
    	       device[d].DevResults,
    	       sizeof(int)*device[d].ncell*device[d].NumberOfTemperatures,
    	       cudaMemcpyDeviceToDevice);
  }
  
  Reduce<int>(1, NumberOfTemperatures, device, Ratio);

  for(auto T=0;T<device.size()*device[0].NumberOfTemperatures;T++) {
    Ratio[T]*=1.0/((double)(device[0].nspins*McSteps));
  }
}
#if  !defined(DISCRETE)


void CalculateSolidAngle(std::vector<dev> &device,
			 const int OverRelaxation,
			 const std::vector<int> &McSteps,
			 const TYPE ratio,
			 vector<double> &Temperatures,
			 vector<TYPE> &CosThetaSolid)
{
  TYPE calculated_ratio = 0;
  TYPE *RatioTable = (TYPE *) malloc(sizeof(TYPE)*Temperatures.size());
  TYPE *ThetaSolid = (TYPE *) malloc(sizeof(TYPE)*Temperatures.size());
  int loop = 0;
  MonteCarlo(device,
	     OverRelaxation,
	     1,
	     McSteps,
	     Temperatures,
	     CosThetaSolid,
	     RatioTable);

  SynchronizeJobs(device);
  CalculateRatio(Temperatures.size(), McSteps[0], device, RatioTable);
  
  for(auto T=0;T<Temperatures.size();T++) {
    ThetaSolid[T] = acos(CosThetaSolid[T]);
  }

  printf("%.10lf %6lf %.6lf %.6lf\n", CosThetaSolid[0], calculated_ratio, RatioTable[0], ThetaSolid[0]); 

  while((calculated_ratio < ratio)&&(loop<1000)) {
    MonteCarlo(device,
	       OverRelaxation,
	       1,
	       McSteps,
	       Temperatures,
	       CosThetaSolid,
	       RatioTable);

    SynchronizeJobs(device);
    CalculateRatio(Temperatures.size(), McSteps[0], device, RatioTable);
    
    for(auto T=0;T<Temperatures.size();T++) {
      if(RatioTable[T]<ratio) {
	ThetaSolid[T] *= 0.9;
	CosThetaSolid[T] = cos(ThetaSolid[T]);
	calculated_ratio = RatioTable[T];
	if(T==0)
	  printf("%.10lf %6lf %.6lf\n", CosThetaSolid[T], calculated_ratio, RatioTable[T]); 
      }
    }
    if(RatioTable[0]>= ratio)
      break;
    loop++;
  }
  free(RatioTable);
  free(ThetaSolid);
}
#endif

void ParallelTempering(std::vector<dev> &device,
#if   !defined(DISCRETE)
		       const int OverRelaxation,
		       const int ReductionAngle,
#endif
		       const std::vector<int> &McSteps,
		       const int EvenOdd,
		       vector<double> &Temperatures,
#if   !defined(DISCRETE)
		       vector<TYPE> &CosThetaSolid,
#endif
		       vector<double> &Energies)
{  
  MonteCarlo(device,
#if   !defined(DISCRETE)
	     OverRelaxation,
	     ReductionAngle,
#endif
	     McSteps,
#if   !defined(DISCRETE)
	     Temperatures,
	     CosThetaSolid, 
	     NULL
#else
	     Temperatures
#endif
	     );
  SynchronizeJobs(device);

  ExtractConfigurationsFromMonteCarlo(device);

  TotalEnergyv2(device, 
		Energies);
  
  TrySwapingTemperatures(
#if   !defined(DISCRETE)
			 CosThetaSolid,
#endif
			 Energies,
			 Temperatures,
			 EvenOdd);
}


#if   !defined(DISCRETE)
void SimulatedAnnealingWithReductionAngle(std::vector<dev> &device,
					  const int CosThetaSolidAlreadyCalculated,
					  const int SimulatedAnnealingSteps,
					  const int OverRelaxation,
					  const int ReductionAngle,
					  const vector<int> &McSteps,
					  vector<double> &Temperatures,
					  vector<TYPE> &CosThetaSolid,
					  vector<TYPE> &Energies)
{
  double StartingTemp = 10.0;
  vector<double> Temp(Temperatures.size());
  vector<double> ratio(Temperatures.size());
  vector<TYPE> TempCosTheta(Temperatures.size());
  vector<int> TempMcSteps(Temperatures.size());

  
  for(auto d=0;d<device.size();d++) {
    cudaSetDevice(d);
    GenerateNewRandomNumbers(&device[d], d, 4*device[d].nspins, device[d].NumberOfTemperatures);
    
    Generate3DVectorList<<<device[d].nspins*device[d].NumberOfTemperatures/64, 64>>>(device[d].RngDev,
										     device[d].nspins*device[d].NumberOfTemperatures,
										     device[d].DevSpins);
  }

  for(auto j=0;j<Temperatures.size();j++)  {
    Temp[j] = Temperatures[0];
    TempCosTheta[j] = -1.0;
  }

  for(auto T=0;T<Temperatures.size();T++) { // Loop over the temperatures    

    // loop over the systems. All systems are at the same temperature
    for(auto sys=0;sys<Temperatures.size();sys++) {
      Temp[sys] = Temperatures[T];
      TempMcSteps[sys] = 25;
      if(CosThetaSolidAlreadyCalculated)
	TempCosTheta[sys] = CosThetaSolid[T];
      else  {
	if(T!=0) {
	  TempCosTheta[sys] = CosThetaSolid[T-1];
	  StartingTemp = Temperatures[T-1];
	}
      }
    } 

    SimulatedAnnealing(device,
		       0,
		       SimulatedAnnealingSteps,
		       OverRelaxation,
		       ReductionAngle,
		       TempMcSteps,
		       Temp,
		       StartingTemp,
		       TempCosTheta);    
    
    for(auto j=0;j<Temperatures.size();j++) 
      TempMcSteps[j] = 200000;

    // Do the monte carlo after the optimal angle is calculated
    MonteCarlo(device,
	       OverRelaxation,
	       ReductionAngle,
	       TempMcSteps,
	       Temp,
	       TempCosTheta, 
	       NULL);

    // Calculate the solid angle if not done already
    if(!CosThetaSolidAlreadyCalculated) {
      SynchronizeJobs(device);      
      for(auto j=0;j<Temperatures.size();j++) 
	TempMcSteps[j] = 2500;
	
      CalculateSolidAngle(device,
			  OverRelaxation,
			  TempMcSteps,
			  0.4,
			  Temp,
			  TempCosTheta);
      
      CosThetaSolid[T] = TempCosTheta[0];
    }

    for(auto j=0;j<Temperatures.size();j++) 
      TempMcSteps[j] = 200000;
    
    // Do the monte carlo after the optimal angle is calculated
    MonteCarlo(device,
	       OverRelaxation,
	       ReductionAngle,
	       TempMcSteps,
	       Temp,
	       TempCosTheta, 
	       NULL);
  }
    
  CleanUp(device);
  Temp.clear();
  TempCosTheta.clear();
  ratio.clear();
  TempMcSteps.clear();
}
#endif
#ifdef DISCRETE
int ApplyParallelTemperingDuringMeasure(int Apply, 
					const std::vector<int> &McStepsTable, 
					std::vector<double> &Temperatures, 
					std::vector<dev> &Device)
{
  if(!Apply) {
    // can do monte carlo while searching for all loops
    MonteCarlo(Device,
	       McStepsTable,
	       Temperatures);
    return 0;
  }
  
  std::vector<double> Energies(Temperatures.size());
  std::vector<int> McStepsTable2(Temperatures.size());
  TotalEnergyv2(Device, Energies);
  TrySwapingTemperatures(Energies,
			 Temperatures,
			 Apply%2);

  
  for(auto T=0;T<Temperatures.size();T++) {
    McStepsTable2[T] = 500;
  }

   MonteCarlo(Device,
	      McStepsTable2,
	      Temperatures);
   Energies.clear();
   McStepsTable2.clear();
   return Apply++;
}
#else
int ApplyParallelTemperingDuringMeasure(int Apply, 
					const int _OverRelaxation,
					const int ReduceAngle,
					const std::vector<int> &McStepsTable, 
					std::vector<double> &CosThetaSolid,
					std::vector<double> &Temperatures, 
					std::vector<dev> &Device)
{
   if(!Apply) {
    // can do monte carlo while searching for all loops
     MonteCarlo(Device,
		_OverRelaxation,
		ReduceAngle,
		McStepsTable,
		Temperatures,
		CosThetaSolid,
		NULL);
   }
   
   std::vector<double> Energies(Temperatures.size());
  std::vector<int> McStepsTable2(Temperatures.size());
  
  TotalEnergyv2(Device, Energies);

  TrySwapingTemperatures(CosThetaSolid,
			 Energies,
			 Temperatures,
			 Apply%2);

  for(auto T=0;T<Temperatures.size();T++) {
    McStepsTable2[T] = 500;
  }

  MonteCarlo(Device,
	     _OverRelaxation,
	     ReduceAngle,
	     McStepsTable2,
	     Temperatures,
	     CosThetaSolid,
	     NULL);
  Energies.clear();
  McStepsTable2.clear();
  return Apply++;
  
}  
#endif