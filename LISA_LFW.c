//  -=================================================-
//   lisa09.c  +++  Spectrum estimation via Gibbs step
//  ---------------------------------------------------
//            !!  PARALLEL IMPLEMENTATION  !!          
//                     ...3rd try...
//           Parallel chains now sampling from 
//       simple, regular posterior distributions...
//  -=================================================-
// Whittle's Likelihood With Subtotal approach
//// Staaaaaaaaaaaaaaaaaaaaaaaaaaaaaaart
// This is using psdpriordf... but no chissq(df) and sigmasq2spec= christian

// Low Frequency Approach (It does not have TDIgenerator function)
// Here mich must always be set as true and TruncEcc and can either true or false according to your needs 


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <complex.h>
#include <fftw3.h>   /*  see: www.fftw.org  */
#include "readxml.h"
#include "ezxml.h"
#include "lisaxml.h"
#include <randlib.h>
#include <time.h>
#include <mpi.h>

#define LENGTH_OF_CHOPPED 17

const double pi     = 3.1415926535897932385; //rand()%(6 + 1 - 4)+4;
const double twopi  = 6.2831853071795864769; //rand()%(6 + 1 - 4)+4;
const double halfpi = 1.5707963267948966192;
const double c      = 2.99792458e8;       // inherited from `codewalk.c'
const double G      = 6.67259e-11;        //   "
const double AU     = 1.49597870660e11;   //   "
const double year   = 3.15581498e7;       //   "
const double SOLARMASSINSEC = 4.92549095e-6;
//const double Msun   = 1.9889e30;          //   "
//const double Mpc    = 3.0856675807e22;    //   "

const double Msun   = 1.98892e30;         //   "
const double Mpc    = 3.0856775807e22;    //   "
const double GPC    = 3.08568025e25;
const double GPCINSEC = 1.02927139014e14;// 14 changed 4m 17 if Mpcinsec


const double freqBandLower = 5e-5;
const double freqBandUpper = 0.01;
/* define indices of elements of parameter vector: */
/* general:  */
const long lognuz = 0; //log of orbital frequency in Hz at t=0
const long logmu = 1; //log of small black hole mass in es
const long logM = 2; //log of central black hole mass in solar masses
const long ez = 3; //Eccentricity of orbit at t=0
const long gamz = 4; //Periholion precession phase at t=0
const long phiz = 5; //Orbital phase at t=0
const long colati = 6; //Colatitude of source
const long longi = 7; //Longitude of source
const long Lambda = 8; //Inclination of orbit
const long alpz = 9; //Orbital plane precession phase at t=0
const long SMBHSpin = 10; //Black hole spin
const long thetaSpin = 11; //Orientation of black hole spin in ssb - colatitude
const long phiSpin = 12; //Orientation of black hole spin in ssb - azimuth
const long logDl = 13; // log of luminosity distance in Gpc
const int NParams=14;

/* degrees-of-freedom parameter specifying spectrum's prior weight: */
const double PSDpriorDF    = 2.0;
const double prior_reference_mt = 2.0e6;    // Inspiral of 2 x 1 Mio Sun mass...
const double prior_dist_90      = 100.0; // ...detectable to 100 thousand...
const double prior_dist_10      = 110.0; // ...and 110 thousand Mpc distance.

/* flag indicating common prior (or not) for A and E spectra: */
const int AEcommonSpectrum = 1;
const int AEcommonPrior    = 1;
const int rate =1;

/* Adjustment of Overall Scale*/
double adj = 0.01;// from .02 to .01 by waqas
/*Number of iterations required to compute PropCov matrix*/	
int datasize =10000;
int datastart=10000;
int dataend =20000;						
/* proposal covariance matrix (off-diagonal elements specified later): */			
double PropCov[14][14]={{                                           1.0e-7,    .0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0}, // lognuz
                       {.0,                                         1.0e-3,     .0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0}, // logmu
                        {.0,.0,                                     1.0e-8,      .0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0}, // logM
                        {.0,.0,.0,                                  2.0e-5,       .0,.0,.0,.0,.0,.0,.0,.0,.0,.0}, //	ez  ------------
                        {.0,.0,.0,.0,                                .03,          .0,.0,.0,.0,.0,.0,.0,.0,.0}, // gamz
                        {.0,.0,.0,.0,.0,                             .01,          .0,.0,.0,.0,.0,.0,.0,.0}, // phiz  
                        {.0,.0,.0,.0,.0,.0,                          .01,           .0,.0,.0,.0,.0,.0,.0}, // colati
                        {.0,.0,.0,.0,.0,.0,.0,                       .03,             .0,.0,.0,.0,.0,.0}, // longi  
                        {.0,.0,.0,.0,.0,.0,.0,.0,                   1.0e-3,        .0,.0,.0,.0,.0}, // Lambda
                        {.0,.0,.0,.0,.0,.0,.0,.0,.0,                 .03,               .0,.0,.0,.0}, // alpz
                        {.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,             1.0e-5,          .0,.0,.0}, // Spin
                        {.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,           .03,                 .0,.0}, // ThetaSpin
                        {.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,        .01,                  .0}, // PhiSpin
                        {.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,   1.0e-2,                }}; // LogDl
double MVNpar[120];
char datapath[]   = "data";
char noisefile1[]   = "challenge1B.3.2-training-strain.xml";
char datafile[]   = "lisa-noise.xml";
char outputpath[] = "output";
char outputfile[] = "NGC_221_2020_2_n"; //  FFTW_ESTIMATE and logM changed....

/*-- definition of `TDIframework' structure: --*/
typedef struct {
  /* elements referring to data and TDIgenerator output:                                      */
  double       dataStart;     /* time point of first sample in the data (seconds)             */
  double       dataDeltaT;    /* time resolution of the data (aka `cadence', `dt')            */
  long         dataN;         /* number of samples in data (former `Nsec')                    */
  /* elements referring to internally processed signal (padded & at higher resolution):       */
  double       padding;       /* padding at both ends of the data, in seconds                 */
  long         dataNpadded;   /* number of samples in data plus padding (`N')                 */
  double       internalStart; /* time point of first sample of internally processed signal    */
  double       internalDeltaT;/* time resolution for internal computations                    */
  long         internalN;     /* number of samples for internally processed signal (`Ninsec') */
  /* LISA model details:                                                                      */
  double       kappa;         /* initial azimuthal position in the ecliptic plane             */
  double       lambda;        /* initial orientation of the LISA spacecraft                   */
  double       orbitRadius;   /* orbital radius of the guiding centre                         */
  double       armLength;     /* mean arm length of the LISA detector (metres)                */
  double       eccent;        /* eccentricity of the LISA spacecraft orbits                   */
  /* FT-related elements:                                                                     */
  double       *FTin;         /* (internal) FT input vector (of length `dataN')               */
  double       *FTwin;        /* (internal) FT window vector (of length `dataN')              */
  double       FTwss;         /* (internal) sum of squared windowing coefficients             */
  fftw_complex *FTout;        /* (internal) FT output vector                                  */
  long         FToutLength;   /* length of FT output vector                                   */
  double       *FTfreq;       /* frequencies for `FTout', `noisePSD', `AdataFT' & `EdataFT'   */
  double       FTnyquist;     /* FT Nyquist frequency (=FTfreq[FToutLength-1]), in Hz         */
  double       deltaFT;       /* length of FT'd data (in seconds)                             */
  fftw_plan    FTplan;        /* Transform plan (only used internally--access via `FTexec()') */
  fftw_complex *AdataFT;      /* Fourier transform of data (A)                                */
  fftw_complex *EdataFT;      /* Fourier transform of data (E)                                */
  double       *AresidualPro; /* vector of A's (freq.-domain) residuals (sum of squared) prop-*/
  double       *EresidualPro; /* vector of E's (freq.-domain) residuals (      "       ) osed.*/
  double       *AresidualAcc; /* vector of A's (freq.-domain) residuals (      "       ) acce-*/
  double       *EresidualAcc; /* vector of E's (freq.-domain) residuals (      "       ) pted.*/
  double       *Asigma;  /* vector for A's sigma prior                                   */
  double       *Esigma;  /* vector for E's sigma prior                                   */
  double       *AnoisePSD;    /* vector for A's (log-) noise Power Spectral Density           */
  double       *EnoisePSD;    /* vector for E's (log-) noise Power Spectral Density           */
} TDIframework;


/*-- define function prototypes: --*/

  void TDIcombi(TimeSeries *ts, int addA, int addE, int addT);
  void chopTimeSeries(TimeSeries *ts, double from, double to);
  void fixTimeSeries(TimeSeries *ts);
  void noisePSDinit(TDIframework *fw, //char *noisefilename, 
                    double start, double end);
  void TDIinit(TDIframework *fw, TimeSeries *ts, int frequencydomain);
  void freeTDIframework(TDIframework *fw);
  void FTexec(TDIframework *fw, double *input, fftw_complex *output);

  void GenBCWaveStart(TDIframework *fw, double *par,
                               double *pluswave, double *crosswave);

double logprior(double *parameter, TDIframework *fw);
  void generateresiduals(TDIframework *fw, double *parameter);
  //double logposterior(TDIframework *fw, double *theta, double lprior);
double signaltonoiseratio(TDIframework *fw, double *parameter);
double tukey(int j, int N, double r);
double genprop(double scale);
  void propose(double *state, double *new, double temperature);
void drawspectrum(TDIframework *fw, double temperature);
double likelihood(TDIframework *fw, double *AresiVec, double *EresiVec);
  void metropolis(TDIframework *fw, double startpar[14], long iterations, 
                  int MPIsize, int MPIrank);
  void printtime();
  int scalelow, scalehigh, switchi;

//void covmat(double *data, int datasize, double temperature);


/*-- define functions: --*/

void TDIcombi(TimeSeries *ts, int addA, int addE, int addT)
/***********************************************************/
/* Compute `optimal' TDI combinations A/E/T from X/Y/Z     */
/* following Prince et al (2002), Phys. Rev. D 66:122002.  */
/*   A := (Z - X) / sqrt(2)                                */
/*   E := (X - 2*Y + Z) / sqrt(6)                          */
/*   T := (X + Y + Z) / sqrt(3)                            */
/* for now actually following Cornish's implementation:    */
/*   A := (2*X - Y - Z) / 3                                */
/*   E := (Z - X) / sqrt(3)                                */
/*  (T := ...?)                                            */
/* Given the original TimeSeries object `ts', additional   */
/* columns are added for requested derivatives.            */
/* Flags `addA', `addE' and `addT' (0 or 1) indicate which */
/* of these are supposed to be added.                      */
/* FOR NOW THE DATA ARE ASSUMED TO HAVE 4 COLUMNS IN THE   */
/* FOLLOWING ORDER: t, X, Y, Z (!!)                        */
/* In future the code might recognise the columns by their */
/* names, but the names would then need to be somewhat     */
/* standardised...                                         */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* EXAMPLE:                                                */
/*   TimeSeries *ts = NULL;                                */
/*   ts = getTDIdata("example.xml");                       */
/*   TDIcombi(ts, 1, 1, 0);                                */
/***********************************************************/
{
  long       Xindex=1;      // index of `X' column
  long       Yindex=2;      // index of `Y' column
  long       Zindex=3;      // index of `Z' column
  int        namelength=10; // maximum length of (new) columns' names
  double     sqrt3 = sqrt(3.0);
  // double     sqrt2 = sqrt(2.0);
  // double     sqrt6 = sqrt(6.0);
  long       numberofcolumns   = ts->Records;
  long       additionalcolumns = addA + addE + addT;
  DataColumn **newdatapointer=NULL;
  char       *oldname;
  long       i,j;
  if (!(((addA==1)||(addA==0)) & ((addE==1)||(addE==0)) & ((addT==1)||(addT==0))))
    printf("  !! ERROR: invalid arguments in TDIcombi() !!\n");
  /* set up new data pointer to old & additional data columns: */
  newdatapointer = (DataColumn**) malloc((numberofcolumns+additionalcolumns) * sizeof(DataColumn*));
  for (i=0; i<numberofcolumns; ++i) 
    newdatapointer[i] = ts->Data[i];
  for (i=numberofcolumns; i<numberofcolumns+additionalcolumns; ++i){
    newdatapointer[i] = (DataColumn*) malloc(sizeof(DataColumn));
    newdatapointer[i]->Name       = (char*) malloc((namelength+1) * sizeof(char));
    newdatapointer[i]->TimeOffset = ts->TimeOffset;
    newdatapointer[i]->Cadence    = ts->Cadence;
    newdatapointer[i]->Length     = ts->Length;
    newdatapointer[i]->data       = (double*) malloc(ts->Length * sizeof(double));
  }
  i = numberofcolumns;
  if (addA){ /*-- add the `A' column: --*/
    for (j=0; j<newdatapointer[i]->Length; ++j)

      newdatapointer[i]->data[j] = (2.0*newdatapointer[Xindex]->data[j]
                                    - newdatapointer[Yindex]->data[j]
                                    - newdatapointer[Zindex]->data[j]) / 3.0;
    sprintf(newdatapointer[i]->Name, "%s", "A");
    ++i;
  }
  if (addE){ /*-- add the `E' column: --*/
    for (j=0; j<newdatapointer[i]->Length; ++j)

      newdatapointer[i]->data[j] = (newdatapointer[Zindex]->data[j]
                                    - newdatapointer[Xindex]->data[j]) / sqrt3;
    sprintf(newdatapointer[i]->Name, "%s", "E");
    ++i;
  }
  if (addT){ /*-- add the `T' column: --*/
    for (j=0; j<newdatapointer[i]->Length; ++j)
      newdatapointer[i]->data[j] = (newdatapointer[Xindex]->data[j]
                                    + newdatapointer[Yindex]->data[j]
                                    + newdatapointer[Zindex]->data[j]) / sqrt3; 
    sprintf(newdatapointer[i]->Name, "%s", "T");
    ++i;
  }
  /* update the actual TimeSeries object (`ts'): */
  free(ts->Data);
  ts->Data = newdatapointer;
  ts->Records += additionalcolumns;
  i=0; 
  while (ts->Name[i] != '\0') ++i;
  oldname = ts->Name;
  ts->Name = (char*) malloc(i+(additionalcolumns*(namelength+1))+1);
  for (j=0; j<=i; j++)
    ts->Name[j] = oldname[j];
  free(oldname);
  for (i=numberofcolumns; i<numberofcolumns+additionalcolumns; ++i){
    sprintf(ts->Name, "%s,%s", ts->Name, ts->Data[i]->Name);
  }
}


void chopTimeSeries(TimeSeries *ts, double from, double to)
/***********************************************************/
/* Chop a section out of the `ts'.                         */
/* Range is defined by `to' and `from'.                    */
/* Note that you may also use the C constant `HUGE_VAL'    */
/* (and `-HUGE_VAL') as limits.                            */
/* e.g.:     chopTimeSeries(ts, -HUGE_VAL, 1000.0)         */
/*       or  chopTimeSeries(ts, 1000.0, HUGE_VAL)          */
/* should return the time series up to 1000.0 and          */
/* after 1000.0.                                           */
/* `TimeOffset' and time stamps (1st data column) are      */
/* simply copied, and not changed!                         */
/* The time corresponding to each sample (row) is assumed  */
/* to be given by the first column (plus `TimeOffset').    */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* EXAMPLE:                                                */
/*   TimeSeries *ts = NULL;                                */
/*   ts = getTDIdata("example.xml");                       */
/*   chopTimeSeries(ts, 3840.0, HUGE_VAL);                 */
/***********************************************************/
{
//printf("\n--> chopTimeSeries(ts, %f, %f)\n", from, to);
  long tindex = 0;  // index of `t' column
  long   firstindex, lastindex, newLength, i, j;
  double *newcolumn;
  i=0;
  while ((ts->TimeOffset + ts->Data[tindex]->data[i]) < from) ++i;
  firstindex = i;
  if (to >= (ts->TimeOffset + ts->Data[tindex]->data[ts->Length-1]))
    lastindex = ts->Length-1;
  else {
    while ((i<ts->Length) && ((ts->TimeOffset + ts->Data[tindex]->data[i]) <= to)) ++i;
    lastindex = i-1;
  }
  newLength = lastindex - firstindex + 1;
  for (i=0; i<ts->Records; ++i){
    newcolumn = (double*) malloc(newLength * sizeof(double));
    for (j=0; j<newLength; ++j)
      newcolumn[j] = ts->Data[i]->data[firstindex+j];
    free(ts->Data[i]->data);
    ts->Data[i]->data = newcolumn;
    ts->Data[i]->Length = newLength;
  }
  ts->Length = newLength;

}


void fixTimeSeries(TimeSeries *ts)
/***********************************************/
/* A quick-and-dirty function to `fix' the     */
/* broken timestamps of a `TimeSeries' object. */
/* * * * * * * * * * * * * * * * * * * * * * * */
/* EXAMPLE:                                    */
/*   TimeSeries *ts = NULL;                    */
/*   ts = getTDIdata("example.xml");           */
/*   fixTimeSeries(ts);                        */
/***********************************************/
{
  long tindex = 0;  // index of `t' column
  long i;
  for (i=0; i<ts->Length; ++i){
    ts->Data[tindex]->data[i] = ts->Cadence/2.0 + ((double)i)*ts->Cadence;
  }
}


void noisePSDinit(TDIframework *fw, double start, double end)
/*******************************************************************/
/* Estimates the noise PSD from the file given by `noisefilename'  */
/* (assuming that file contains only noise, no signal).            */
/* See also P.D. Welch, IEEE Transactions Audio Electroacoust.     */
/* AU-15(2):70-73, 1967.                                           */
/* Overlapping segments of the data are individually FT'd and the  */
/* resulting spectra are averaged.                                 */
/* The two spectra (by now seperately for A & E) are interpolated  */
/* to the frequency grid given by `fw->FTfreq' and stored under    */
/* `fw->AnoisePSD' and `fw->EnoisePSD'.                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* This function is (by now) called within `TDIinit()'.            */
/*******************************************************************/
{
  //long segmentlength = 65536; // = 2^16
  long segmentlength = 2048; // = 2^15
  // (Number of samples in each data segment (must be even!). 1 year ~ 2^21)
  long K; // (Number of overlapping segments of above length.)
  long PSDlength     = (segmentlength/2)+1; // length of FT output
  double *FourierIn;
  fftw_complex *FourierOut;
  fftw_plan FourierPlan;
  double *PSD;         // actual PSD estimate
  double *win;         // windowing coefficients ()
  double wss=0.0;      // sum of squared windowing coefs
  TimeSeries *noisefile;
  long Aindex, Eindex;
  double log2 = log(2.0);
  // double twopi = 2.0*3.141592653589793;
  double dblindex;
  long lowindex, highindex;
  double weight1, weight2;
  long i,j;
  double sigmasq2spec = 2.0*log(fw->dataDeltaT) + log(fw->dataN);
  /* initialise stuff: */
  //printf(" | estimating A & E noise spectrum...\n");
  FourierIn = (double*) fftw_malloc(segmentlength * sizeof(double));
  FourierOut = (fftw_complex*) fftw_malloc(PSDlength*sizeof(fftw_complex));
  FourierPlan = fftw_plan_dft_r2c_1d(segmentlength, FourierIn, FourierOut, FFTW_ESTIMATE);
  PSD = (double*) malloc(PSDlength*sizeof(double));

  char completefilename[200];
  sprintf(completefilename, "%s/%s", datapath, datafile);
  //fprintf(stderr," | getTDIdata(): ");
  noisefile = getTDIdata(completefilename);
  fixTimeSeries(noisefile);
  chopTimeSeries(noisefile, start, end);

  /* construct A/E channels: */
  TDIcombi(noisefile, 1, 1, 0);

  /* for now assume A and E were in channels #4 and #5: */
  Aindex = 4;
  Eindex = 5;
  K = (noisefile->Length / (segmentlength/2)) - 1; // number of overlapping segments
  /* windowing coefficients and their sum (Hann window): */
  win = (double*) malloc(segmentlength * sizeof(double));
  for (j=0; j<segmentlength; j++){
   win[j] = 0.5*(1.0-cos(((double)j/(double)segmentlength)*twopi));
    //win[j] = 1.0;
    wss += win[j]*win[j];
  }
  /*  Already wrap further coefficients into `win' so they    */
  /*  don't need to be applied to the FT'd values afterwards  */
  /*  (this can be done before FTing since FT is linear):     */
  for (j=0; j<segmentlength; ++j) {
    //  former bigged version:
    //  win[j] *= sqrt( ((double)noisefile->Cadence) / (wss * ((double) K)));
    //  new version:
    win[j] *= ((double) noisefile->Cadence) / sqrt(wss * ((double) K));
  }

  /*-- noise PSD for `A' --*/
  for (j=0; j<PSDlength; ++j)
    PSD[j] = 0.0;
  for (i=0; i<K; ++i){
    for (j=0; j<segmentlength; ++j)
      FourierIn[j] = win[j] * noisefile->Data[Aindex]->data[j+i*(segmentlength/2)];
    fftw_execute(FourierPlan);
    for (j=0; j<PSDlength; ++j)
      PSD[j] += pow(cabs(FourierOut[j]), 2.0);
  }
  /* log of squared PSD: */
  for (j=0; j<PSDlength; ++j)
    PSD[j] = log(PSD[j]) + log2;
  /* interpolate to same grid as other data FTs: */
  fw->Asigma = (double*) malloc(fw->FToutLength * sizeof(double));
  for (i=0; i<fw->FToutLength; ++i){
    dblindex = (((double)i)/((double)(fw->FToutLength-1)))*((double)(PSDlength-1));
    lowindex = (long) floor(dblindex); /* (truncated!) */
    highindex = lowindex + 1;
    weight1   = ((double)highindex) - dblindex;
    weight2   = dblindex - ((double)lowindex);
    fw->Asigma[i] = weight1*PSD[lowindex] + weight2*PSD[highindex];
  }

  /*-- noise PSD for `E' --*/
  for (j=0; j<PSDlength; ++j)
    PSD[j] = 0.0;
  for (i=0; i<K; ++i){
    for (j=0; j<segmentlength; ++j)
      FourierIn[j] = win[j] * noisefile->Data[Eindex]->data[j+i*(segmentlength/2)];
    fftw_execute(FourierPlan);
    for (j=0; j<PSDlength; ++j)
      PSD[j] += pow(cabs(FourierOut[j]), 2.0);
  }
  /* log of squared PSD: */
  for (j=0; j<PSDlength; ++j)
    PSD[j] = log(PSD[j]) + log2;
  /* interpolate to same grid as other data FTs: */
  fw->Esigma = (double*) malloc(fw->FToutLength * sizeof(double));
  for (i=0; i<fw->FToutLength; ++i){
    dblindex = (((double)i)/((double)(fw->FToutLength-1)))*((double)(PSDlength-1));
    lowindex = (long) floor(dblindex); /* (truncated!) */
    highindex = lowindex + 1;
    weight1   = ((double)highindex) - dblindex;
    weight2   = dblindex - ((double)lowindex);
    fw->Esigma[i] = weight1*PSD[lowindex] + weight2*PSD[highindex];
  }

  /* if requested, pool both A & E prior spectra: */
  if (AEcommonPrior) {
    for (i=0; i<fw->FToutLength; ++i){
      fw->Asigma[i] += fw->Esigma[i];
      fw->Asigma[i] /= 2.0;
      fw->Esigma[i] = fw->Asigma[i];
    } 
  }

  /* `un-log' the PRIOR PSDs and transform to `sigma' scale: */
  for (i=0; i<fw->FToutLength; ++i){
    fw->Asigma[i] = exp(fw->Asigma[i] - sigmasq2spec);
    fw->Esigma[i] = exp(fw->Esigma[i] - sigmasq2spec);
  } 

  fftw_free(FourierIn);
  fftw_free(FourierOut);
  fftw_destroy_plan(FourierPlan);
  free(win);
  free(PSD);
  freeTimeSeries(noisefile);
}


void TDIinit(TDIframework *fw, TimeSeries *ts, int frequencydomain)
/***************************************************************/
/* Initialises the elements of `fw'.                           */
/* The flag `frequencydomain' indicates whether the elements   */
/* related to frequency-domain likelihood computations are     */
/* supposed to be initialised as well.                         */
/* (Unforfunately the code will still require FFTW to be in-   */
/* stalled, even if it's not actually used ... it'll probably  */
/* take a lot of manual commenting-out to still get it         */
/* running...)                                                 */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* EXAMPLE:                                                    */
/*   TimeSeries *ts = NULL;                                    */
/*   TDIframework  tdiframe;                                   */
/*   ts = getTDIdata("example.xml");                           */
/*   TDIinit(&tdiframe, ts);                                   */
/***************************************************************/
{
  long i, root;
  long tindex=0;  // index of time (`t') column in TimeSeries object `ts'
  long Aindex=4;  // index of `A' column in TimeSeries object `ts'
  long Eindex=5;  // index of `E' column in TimeSeries object `ts'
  long extraN;
  double sigmasq2spec = 2.0*log(fw->dataDeltaT) + log(fw->dataN);

  //time_t starttime, endtime;

  fw->kappa  = 0.0;
  fw->lambda = 0.0;
  fw->orbitRadius = AU;  // formerly `Rgc'
  fw->armLength = 5.0e9; // formerly `L'
  fw->eccent = (fw->armLength/(2.0*sqrt(3.0)*fw->orbitRadius)); // formerly `ecc'

  fw->dataStart      = ts->TimeOffset + ts->Data[tindex]->data[0];
  fw->dataDeltaT     = ts->Cadence;
  //if (fw->dataStart != (fw->dataDeltaT/2.0))
  //  printf(" |\n-+-->  CAUTION! Using a non-zero starting point will cause trouble!\n |\n");
  //  //                          (at least by now!)
  fw->dataN          = ts->Length;
  fw->padding        = 900.0;                // seconds. For now hardwired!
  fw->dataNpadded    = fw->dataN + 2*(long)ceil(fw->padding/fw->dataDeltaT);
  fw->internalDeltaT = fw->dataDeltaT / 4.0; // for now hardwired as well!
  fw->internalStart  = fw->dataStart - fw->dataDeltaT * ((long)ceil(fw->padding/fw->dataDeltaT));
  fw->internalN      = 4*fw->dataNpadded - (4-1);


  if (frequencydomain) {
    //printf(" | (also initialising frequency-domain related variables)\n");
    /*-- initialise FT input: --*/
    fw->FTin = (double*) fftw_malloc(fw->dataN * sizeof(double));
    /*-- initialise FT windowing coefficients: --*/
    fw->FTwin = (double*) fftw_malloc(fw->dataN * sizeof(double));
    fw->FTwss = 0.0;
    for (i=0; i<fw->dataN; ++i) {
      //fw->FTwin[i] = 1.0;  // (for `no' windowing = `boxcar' windowing)
      fw->FTwin[i] = tukey(i, fw->dataN, 0.02);
      fw->FTwss += (fw->FTwin[i]*fw->FTwin[i]);
    }
    /*-- initialise FT output: --*/
    fw->FToutLength = (fw->dataN / 2) + 1;
    fw->FTout = (fftw_complex*) fftw_malloc(fw->FToutLength * sizeof(fftw_complex));
    fw->FTfreq = (double*) malloc(fw->FToutLength * sizeof(double));
    for (i=0; i<fw->FToutLength; ++i)
      fw->FTfreq[i] = ((double)i)/((double)(fw->FToutLength)) / (2.0*ts->Cadence);
    fw->FTnyquist = 1.0 / (2.0 * ts->Cadence);
    fw->deltaFT = ts->Cadence * ts->Length;
    /*-- initialise FT plan: --*/
    fw->FTplan = fftw_plan_dft_r2c_1d(fw->dataN, fw->FTin, fw->FTout, FFTW_ESTIMATE);
    //fw->FTplan = fftw_plan_dft_r2c_1d(fw->dataN, fw->FTin, fw->FTout, FFTW_MEASURE);
    //   /!\  caution!
    // when using the `fftw_measure' option, double check whether the results
    // actually stay the same! They once didn't for me, no idea why.
    /*-- FT the data: --*/
    fw->AdataFT = (fftw_complex*) malloc(fw->FToutLength*sizeof(fftw_complex));
    FTexec(fw, ts->Data[Aindex]->data, fw->AdataFT);
    fw->EdataFT = (fftw_complex*) malloc(fw->FToutLength*sizeof(fftw_complex));
    FTexec(fw, ts->Data[Eindex]->data, fw->EdataFT);
    /*-- initialise noise spectrum: --*/
    /* (by now using data file and pretending is was pure noise) */
    // noise PSD for trigger 1 (round 2 blind; 15x2^18s):
    //noisePSDinit(fw, -HUGE_VAL, 604800.0);
	noisePSDinit(fw, -HUGE_VAL, 63072000.0);
    fw->AnoisePSD = (double*) malloc(fw->FToutLength * sizeof(double));
    fw->EnoisePSD = (double*) malloc(fw->FToutLength * sizeof(double));
    // for first likelihood evaluations just copy prior PSDs:
    for (i=0; i<fw->FToutLength; ++i){
      fw->AnoisePSD[i] = log(fw->Asigma[i])+sigmasq2spec;// -log(2.0); // minus extra margin...
      fw->EnoisePSD[i] = log(fw->Esigma[i])+sigmasq2spec;// -log(2.0);
      //fw->AnoisePSD[i] += gennor(-0.7, 1.0);
      //fw->EnoisePSD[i] += gennor(-0.7, 1.0);
    }
    fw->AresidualPro = (double*) malloc(fw->FToutLength * sizeof(double));
    fw->EresidualPro = (double*) malloc(fw->FToutLength * sizeof(double));
    fw->AresidualAcc = (double*) malloc(fw->FToutLength * sizeof(double));
    fw->EresidualAcc = (double*) malloc(fw->FToutLength * sizeof(double));
  
  	root=ceil((double)fw->FToutLength);
    switchi=0;
    for(i=0;i<=root;++i)
      {
	if(fw->FTfreq[i]>=freqBandLower&&switchi==0)
	  {
	    scalelow=i;
	    switchi=1;
	  }
	if(fw->FTfreq[i]>=freqBandUpper&&switchi==1)
	  {
	    scalehigh=i;
	    switchi=2;
	  }
      }
	}  
  else {
    //printf(" | (frequency-domain related variables not initialised)\n");
    fw->FTin         = NULL;
    fw->FToutLength  = 0;
    fw->FTout        = NULL;
    fw->FTfreq       = NULL;
    fw->FTnyquist    = 0.0;
    fw->deltaFT      = 0.0;
    fw->AdataFT      = NULL;
    fw->EdataFT      = NULL;
    fw->AnoisePSD    = NULL;
    fw->EnoisePSD    = NULL;
    fw->Asigma  = NULL;
    fw->Esigma  = NULL;
    fw->AresidualPro = NULL;
    fw->EresidualPro = NULL;
    fw->AresidualAcc = NULL;
    fw->EresidualAcc = NULL;
  }
}



void freeTDIframework(TDIframework *fw)
/* Clean up elements of `fw'. */
{
  fftw_free(fw->FTin);
  fftw_free(fw->FTout); 
  fftw_destroy_plan(fw->FTplan);
  free(fw->FTfreq); 
  free(fw->AdataFT);
  free(fw->EdataFT);
  free(fw->AnoisePSD);
  free(fw->EnoisePSD);
}


void FTexec(TDIframework *fw, double *input, fftw_complex *output)
/***************************************************/
/* Executes a Fourier transform based on           */
/* the `TDIinit()' settings given in `fw'.         */
/* `input'  is assumed to be a vector              */
/*          of length `fw->dataN'.                 */
/* `output' is assumed to be a                     */
/*          pre-allocated (!!) vector              */
/*          of length `fw->FToutLength'.           */
/* `input' is NOT overwritten during FT.           */
/* * * * * * * * * * * * * * * * * * * * * * * * * */
/* EXAMPLE:                                        */
/*   TimeSeries *ts = NULL;                        */
/*   TDIframework  tdiframe;                       */
/*   fftw_complex  *FourierTransform;              */
/*   ts = getTDIdata("example.xml");               */
/*   TDIinit(&tdiframe, ts);                       */
/*   FourierTransform = (fftw_complex*) malloc(tdiframe.FToutLength*sizeof(fftw_complex)); */
/*   FTexec(&tdiframe, ts->Data[4]->data, Aft);    */
/***************************************************/
{
  int i;
  if (output==NULL) 
    printf("  !! ERROR: unallocated output vector in `FTexec()' !!\n");
  // copy AND WINDOW (!) the input data:
  for (i=0; i<fw->dataN; ++i)
    fw->FTin[i] = input[i] * fw->FTwin[i]; // Tukey-2%-window
  fftw_execute(fw->FTplan);
  for (i=0; i<fw->FToutLength; ++i)
    output[i] = fw->FTout[i];
}

void PNevolution(int vlength, double timestep, double *par, double nu0, double *gimdotvec, double *e, double *nu, double *Phi,double *gim, double *alp, int *nwave);

const bool TruncEcc=true;
const int nt = 620000;
const int ndim = 12;
const int modes = 40;

double sh( double f )
{
  double ACC=9.1846875e-52;
  double FLOOR=1.5869625e-41;
  double SHOT=9.1846875e-38;
  double GAL=2.1e-45;
  double EXGAL=4.2e-47;
  double DNDF=2.e-3; 
  double KT = 4.756469e-8;

  bool includeWD=false;
  double sInst;    /* instrumental noise */
  double sGal;     /* Galactic WD confusion noise */
  double sInstGal; /* instrumental plus Galactic WD confusion noise */
  double sExGal;   /* extra-Galactic WD confusion noise */

  sInst = ( ACC*pow( f, -4.0 ) + FLOOR + SHOT*pow( f, 2.0 ));
  if (includeWD) {
          sGal = GAL*pow( f, -7.0/3.0 );
          sExGal = EXGAL*pow( f, -7.0/3.0 );
          sInstGal = sInst/exp( -KT*DNDF*pow( f, -11.0/3.0 ) );
          if ( sInstGal > sInst + sGal )
                  sInstGal = sInst + sGal;
          return sInstGal + sExGal;
  } else
        return sInst;
}

double ArcT(double down, double up) {
  double ArcT;
  ArcT=atan(up/down);
  if (down < 0.) ArcT=ArcT+pi;
  //     if ((up < 0.) && (down > 0)) ArcT=ArcT+2.*pi;
  return ArcT;
}

double J0(double x)
{
  double ax,z;
  double xx,y,ans,ans1,ans2;

  if ((ax=fabs(x)) < 8.0) {
    y=x*x;
    ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
                                            +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
    ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
                                          +y*(59272.64853+y*(267.8532712+y*1.0))));
    ans=ans1/ans2;
  } else {
    z=8.0/ax;
    y=z*z;
    xx=ax-0.785398164;
    ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
                                    +y*(-0.2073370639e-5+y*0.2093887211e-6)));
    ans2 = -0.1562499995e-1+y*(0.1430488765e-3
                               +y*(-0.6911147651e-5+y*(0.7621095161e-6
                                                       -y*0.934935152e-7)));
    ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
  }
  return ans;
}

/* ************************************************************
   FUNCTION J1
   ************************************************************ */

double J1(double x)
{
  double ax,z;
  double xx,y,ans,ans1,ans2;

  if ((ax=fabs(x)) < 8.0) {
    y=x*x;
    ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
                                              +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
    ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
                                           +y*(99447.43394+y*(376.9991397+y*1.0))));
    ans=ans1/ans2;
  } else {
    z=8.0/ax;
    y=z*z;
    xx=ax-2.356194491;
    ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
                               +y*(0.2457520174e-5+y*(-0.240337019e-6))));
    ans2=0.04687499995+y*(-0.2002690873e-3
                          +y*(0.8449199096e-5+y*(-0.88228987e-6
                                                 +y*0.105787412e-6)));
    ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    if (x < 0.0) ans = -ans;
  }
  return ans;
}

/* ************************************************************
   FUNCTION Jn
   ************************************************************ */

double Jn(int n, double x) {
  const int IACC=160;
  const double BIGNO=1.e10;
  const double BIGNI=1.e-10;
  int j,jsum,m;
  double ax,bj,bjm,bjp,sum,tox,Jnval;
  if(n == 0)
    return J0(x);
  if(n == 1)
    return J1(x);
  ax=fabs(x);
  if(ax == 0.)
    Jnval=0.;
  else if(ax > ((double)n)) {
    tox=2./ax;
    bjm=J0(ax);
    bj=J1(ax);
    for(j=1;j<n;j++) {
      bjp=j*tox*bj-bjm;
      bjm=bj;
      bj=bjp;
    }
    Jnval=bj;
  } else {
    tox=2./ax;
    m=2*((n+((int)(sqrt((double)(IACC*n)))))/2);
    Jnval=0.;
    jsum=0;
    sum=0.;
    bjp=0.;
    bj=1.;
    for (j=m;j>0;j--) {
      bjm=j*tox*bj-bjp;
      bjp=bj;
      bj=bjm;
      if(fabs(bj) > BIGNO) {
        bj=bj*BIGNI;
        bjp=bjp*BIGNI;
        Jnval*=BIGNI;
        sum*=BIGNI;
      }
      if(jsum != 0)
        sum+=bj;
      jsum=1-jsum;
      if(j == n)
        Jnval = bjp;
    }
    sum=2.*sum-bj;
    Jnval/=sum;
  }
  if((x < 0.) && (n & 1))
    Jnval*=-1.;
  /* **********************************************
     ADDED BY LEOR since NumRec does not treat n=-1
     ********************************************** */
  if(n == -1)
    Jnval*=-1.;
  return Jnval;
}

double Integral(double *f, double timestep, int vlength) {
  int i;
  double integ;

  integ=3./8.*(f[0]+f[vlength])+7./6.*(f[1]+f[vlength-1])+23./24.*(f[2]+f[vlength-2]);

  for(i=3;i<vlength-2;i++)
    integ+=f[i];
  integ*=timestep;
  return integ;
}

void waveform(double tstart,double *par, double nu0, int vlength, double timestep, double *hI, double *hII, int nmodes, double zeta, int WCN) {
  int i,i0,n;
  double t0,mu,M,qS,phiS,lam,S,qK,phiK,Sn;
  double fn,invsqrtS[modes+1];
  double e,nu,Phi,gim,alp,t,ne,nPhi;
  double a,b,c,An,Amp,Jm2,Jm1,Jm0,Jp1,Jp2,it1,tint;
  double cosqL,sinqL,Ldotn,phiL,BB,CC,Ldotn2;
  double gam2,Aplus,Acros,Sdotn,PhiT,PhiTdot,alpdot;
  double halfsqrt3,prefact,cosq,cosqS,sinqS,coslam,sinlam;
  double cosqK,sinqK,cosalp,sinalp,cos2alp,sin2alp,cosphiK,sinphiK;
  double orbphs,cosorbphs,sinorbphs,phiw,psi,psidown,psiup;
  double cos2phi,sin2phi,cos2psi,sin2psi,cosq1;
  double FplusI,FcrosI,FplusII,FcrosII,AUsec,Doppler;
  double hnI,hnII,cos2gam,sin2gam;
  double betaup,betadown,beta,nfact,cosphiS,sinphiS;

  double *evec,*nuvec,*alpvec,*Phivec,*gimvec,*gimdotvec;
  evec=(double *)malloc((vlength+1)*sizeof(double));
  nuvec=(double *)malloc((vlength+1)*sizeof(double));
  alpvec=(double *)malloc((vlength+1)*sizeof(double));
  Phivec=(double *)malloc((vlength+1)*sizeof(double));
  gimvec=(double *)malloc((vlength+1)*sizeof(double));
  gimdotvec=(double *)malloc((vlength+1)*sizeof(double));

  halfsqrt3=0.5*sqrt(3.);

  double kappa=0.;
  double lambda=0.;
  double phi0=kappa;
  double phiw0=0.75*pi-lambda;

  bool mich=true;
  bool noiseweight=false;

  if (mich)
    prefact=halfsqrt3;
  else
    prefact=1.0;

  AUsec=499.004783702731;

  t0=0.;
  mu=par[logmu];
  M= par[logM];
  cosqS=cos(par[colati]);
  phiS=par[longi];
  coslam=cos(par[Lambda]);
  S=par[SMBHSpin];
  cosqK=cos(par[thetaSpin]);
  phiK=par[phiSpin];

  sinlam=sin(par[Lambda]);

  sinqS=sin(par[colati]);
  sinqK=sin(par[thetaSpin]);
  cosphiK=cos(phiK);
  sinphiK=sin(phiK);
  cosphiS=cos(phiS);
  sinphiS=sin(phiS);

  int nwave;
  PNevolution(vlength,timestep,par,nu0,gimdotvec,evec,nuvec,Phivec,gimvec,alpvec,&nwave);

  
   printf("PT = %i\n", nwave);
  //cout << "Finished PNevolution. Exiting!" << endl;
  //exit(0);
  i0=(int)(t0/timestep);
  PhiT=0.;

  double cX,sX;
  double Apc1, Aps1, Apcn1,psiint,nphiP,Xp[6],e2;
  double Apc2, Aps2, Apcn2;
  double Aqc1, Aqs1, Aqcn1;
  double Aqc2, Aqs2, Aqcn2;
  double Bp1c1,Bp1c2,Bp1s1,Bp1s2,Bp1cn,Bp2c1,Bp2c2,Bp2s1,Bp2s2,Bp2cn,Bc1c1,Bc1c2,Bc1s1,Bc1s2,Bc1cn,Bc2c1,Bc2c2,Bc2s1,Bc2s2,Bc2cn,up,dw;
  double Ap1,Ap2,Ac1,Ac2;

  if (TruncEcc) {
    cX = cosqS*cosqK + sinqS*sinqK*(cosphiS*cosphiK+sinphiS*sinphiK);
    sX = sqrt( sinqS*sinqS*cosqK*cosqK - 2.*sinqS*sinphiS*cosqK*cosqS*sinqK*sinphiK +   cosqS*cosqS*sinqK*sinqK - 2.*cosqS*sinqK*cosphiK*sinqS*cosphiS*cosqK + sinqS*sinqS*cosphiS*cosphiS*sinqK*sinqK*sinphiK*sinphiK - 2.*sinqS*sinqS*cosphiS*sinqK*sinqK*sinphiK*sinphiS*cosphiK + sinqS*sinqS*sinphiS*sinphiS*sinqK*sinqK*cosphiK*cosphiK);

    Apc1 = ( -cosqK*cosphiK*sinqS*cosphiS - cosqK*sinphiK*sinqS*sinphiS + sinqK*cosqS )/(sX);
    Aps1 = ( sinphiK*sinqS*cosphiS - cosphiK*sinqS*sinphiS )/(sX);
    Apcn1 = ( cosqK*cosqS + sinqK*cosphiK*sinqS*cosphiS + sinqK*sinphiK*sinqS*sinphiS - cX)*coslam/(sX*sinlam);

    Apc2 = (sinqS*cosphiS*sinphiK - sinqS*sinphiS*cosphiK )*coslam/(sX);
    Aps2 = ( cosqK*cosphiK*sinqS*cosphiS + cosqK*sinphiK*sinqS*sinphiS - cosqS*sinqK )*coslam/(sX);
    Apcn2 = 0.0;

    Aqc1 = ( sinqS*cosphiS*sinphiK - sinqS*sinphiS*cosphiK  )*cX/(sX);
    Aqs1 = ( cosqK*cosphiK*sinqS*cosphiS + cosqK*sinphiK*sinqS*sinphiS - cosqS*sinqK )*cX/(sX);
    Aqcn1 = 0.0;

    Aqc2 = cX*coslam*( cosqK*cosphiK*sinqS*cosphiS + cosqK*sinphiK*sinqS*sinphiS - sinqK*cosqS)/(sX);
    Aqs2 = -cX*coslam*sinqS*( sinphiK*cosphiS - cosphiK*sinphiS )/(sX);
    Aqcn2 = -( cX*coslam*coslam*( cosqK*cosqS + sinqK*cosphiK*sinqS*cosphiS + sinqK*sinphiK*sinqS*sinphiS ) + 1.- cX*cX - coslam*coslam )/(sX*sinlam);


    Bp1c1 = 2.0*(Apc1*Apcn1 - Aqc1*Aqcn1 + Aqc2*Aqcn2 - Apc2*Apcn2);
    Bp1c2 =  0.5*(Aps2*Aps2 - Aqc1*Aqc1  + Apc1*Apc1  - Aps1*Aps1 + Aqc2*Aqc2 + Aqs1*Aqs1 - Apc2*Apc2 - Aqs2*Aqs2);
    Bp1s1 = 2.0*(Aqs2*Aqcn2 - Aps2*Apcn2 - Aqs1*Aqcn1 + Aps1*Apcn1);
    Bp1s2 = (Apc1*Aps1 + Aqc2*Aqs2 - Apc2*Aps2 - Aqc1*Aqs1);
    Bp1cn = 0.5*(Apc1*Apc1 + Aps1*Aps1 - Aqc1*Aqc1 - Aqs1*Aqs1 - Apc2
                 *Apc2 + Aqc2*Aqc2 + Aqs2*Aqs2 - Aps2*Aps2) + Aqcn2*Aqcn2 - Aqcn1*Aqcn1 + Apcn1*Apcn1 - Apcn2*Apcn2;

    Bp2c1 = (Apcn1*Apc2 + Apc1*Apcn2 - Aqcn1*Aqc2 - Aqc1*Aqcn2);
    Bp2c2 = 0.5*(Aqs1*Aqs2 - Aps1*Aps2 + Apc1*Apc2 - Aqc1*Aqc2);
    Bp2s1 = (Aps1*Apcn2 + Apcn1*Aps2 - Aqcn1*Aqs2 - Aqs1*Aqcn2);
    Bp2s2 = 0.5*( Apc1*Aps2 - Aqc1*Aqs2 + Aps1*Apc2 - Aqs1*Aqc2);
    Bp2cn = 0.5*(Aps1*Aps2 - Aqs1*Aqs2 - Aqc1*Aqc2 + Apc1*Apc2) - Aqcn1*Aqcn2 + Apcn1*Apcn2;

    Bc1c1 = (-Apc2*Aqcn2 - Apcn2*Aqc2 + Apc1*Aqcn1 + Apcn1*Aqc1);
    Bc1c2 = 0.5*( Apc1*Aqc1 - Aps1*Aqs1 - Apc2*Aqc2 + Aps2*Aqs2);
    Bc1s1 = (Apcn1*Aqs1 - Aps2*Aqcn2 + Aps1*Aqcn1 - Apcn2*Aqs2);
    Bc1s2 = 0.5*(-Apc2*Aqs2 + Apc1*Aqs1 - Aps2*Aqc2 + Aps1*Aqc1);
    Bc1cn = -Apcn2*Aqcn2 + Apcn1*Aqcn1 + 0.5*(Apc1*Aqc1 - Aps2*Aqs2 + Aps1*Aqs1 - Apc2*Aqc2);

    Bc2c1 = (Aqc1*Apcn2 + Aqcn1*Apc2 + Apc1*Aqcn2 + Apcn1*Aqc2);
    Bc2c2 = 0.5*( Apc1*Aqc2 - Aps1*Aqs2 + Aqc1*Apc2 - Aqs1*Aps2);
    Bc2s1 = (Apcn1*Aqs2 + Aqs1*Apcn2 + Aps1*Aqcn2 + Aqcn1*Aps2);
    Bc2s2 = 0.5*(Aqc1*Aps2 + Apc1*Aqs2 + Aqs1*Apc2 + Aps1*Aqc2);
    Bc2cn = Aqcn1*Apcn2 + Apcn1*Aqcn2 + 0.5*(Apc1*Aqc2 + Aqs1*Aps2 + Aps1*Aqs2 + Aqc1*Apc2);
  }
  up = (cosqS*sinqK*(cosphiS*cosphiK+sinphiS*sinphiK) - cosqK*sinqS);
  dw = (sinqK*(sinphiS*cosphiK-cosphiS*sinphiK));


  if (dw != 0.0) {
    psiint = ArcT(up, dw); // previously it was psiSL = ArcTan(dw,up)
  }else {
    psiint = halfpi;
  }

  for(i=i0;i<=nwave;i++) {
    /*       If (WCN.eq.0) then
             call NoiseInst (nuvec(i),gimdot0,nmodes,invsqrtS)
             else
             call NoiseFull (nuvec(i),gimdot0,nmodes,invsqrtS)
             endif*/

    hI[i]=0;
    hII[i]=0;

    //     t is actual time (sec) at step i
    t=tstart+timestep*(double)i;

    e=evec[i];
    nu=nuvec[i];
    Phi=Phivec[i];
    gim=gimvec[i];
    alp=alpvec[i];
    cosalp=cos(alp);
    sinalp=sin(alp);
    if (TruncEcc) {
      cos2alp=2.*cosalp*cosalp-1.;
      sin2alp=2.*cosalp*sinalp;
    } else {
      cosqL=cosqK*coslam+sinqK*sinlam*cosalp;
      sinqL=sqrt(1-cosqL*cosqL);
      BB=sinqK*cosphiK*coslam+sinphiK*sinlam*sinalp-cosqK*cosphiK*sinlam*cosalp;
      CC=sinqK*sinphiK*coslam-cosphiK*sinlam*sinalp-cosqK*sinphiK*sinlam*cosalp;
      phiL=ArcT(BB,CC);
      Ldotn=cosqL*cosqS+sinqL*sinqS*cos(phiL-phiS);
      Ldotn2=Ldotn*Ldotn;
    }

    if (mich) {
      orbphs=2.*pi*t/year+phi0;
      cosorbphs=cos(orbphs-phiS);
      sinorbphs=sin(orbphs-phiS);
      cosq=.5*cosqS-halfsqrt3*sinqS*cosorbphs;
      phiw=orbphs+phiw0+ArcT(sinqS*sinorbphs,halfsqrt3*cosqS+.5*sinqS*cosorbphs);
      psiup=.5*cosqK-halfsqrt3*sinqK*cos(orbphs-phiK)-cosq*(cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS));
      psidown=.5*sinqK*sinqS*sin(phiK-phiS)-halfsqrt3*cos(orbphs)*(cosqK*sinqS*sin(phiS)-cosqS*sinqK*sin(phiK))-halfsqrt3*sin(orbphs)*(cosqS*sinqK*cos(phiK)-cosqK*sinqS*cos(phiS));
      psi=ArcT(psidown,psiup);
      cosq1=.5*(1+cosq*cosq);
      cos2phi=cos(2.*phiw);
      sin2phi=sin(2.*phiw);
      cos2psi=cos(2.*psi);
      sin2psi=sin(2.*psi);

      FplusI=cosq1*cos2phi*cos2psi-cosq*sin2phi*sin2psi;
      FcrosI=cosq1*cos2phi*sin2psi+cosq*sin2phi*cos2psi;
      FplusII=cosq1*sin2phi*cos2psi+cosq*cos2phi*sin2psi;
      FcrosII=cosq1*sin2phi*sin2psi-cosq*cos2phi*cos2psi;
    } else {
      FplusI=1.;
      FcrosI=0.;
      FplusII=0.;
      FcrosII=1.;
    }
    /*     --------------------------------------------------------
           Calculate gamma (ext/int mixed) out of gimmel (intrinsic)
           -------------------------------------------------------- */



    Amp=pow((2.*pi*nu*M),2./3.)*zeta;

    if (TruncEcc) {
      gam2=2.*gim;
      cos2gam=cos(gam2);
      sin2gam=sin(gam2);

      Ap1 = Bp1c1*cosalp + Bp1c2*cos2alp + Bp1s1*sinalp+Bp1s2*sin2alp + Bp1cn;
      Ap2 = Bp2c1*cosalp + Bp2c2*cos2alp + Bp2s1*sinalp + Bp2s2*sin2alp + Bp2cn;
      Ac1 = Bc1c1*cosalp + Bc1c2*cos2alp + Bc1s1*sinalp+ Bc1s2*sin2alp + Bc1cn;
      Ac2 = Bc2c1*cosalp + Bc2c2*cos2alp + Bc2s1*sinalp+Bc2s2*sin2alp + Bc2cn;

      e2=e*e;
      Xp[1] = (-3. + 13./8.*e2)*e;
      Xp[2] = 2. - e2*(5. - 23./8.*e2);
      Xp[3] = e*(3. -e2*( 57./8. - 321./64.*e2));
      Xp[4] = e2*(4. - e2*(10. - 101./12.*e2));
      Xp[5] = e*e2*(125./24. - e2*(5375./384. - e2*42125./3072.));
    } else {
      Sdotn=cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS);
      betaup=-Sdotn+coslam*Ldotn;
      betadown=sinqS*sin(phiK-phiS)*sinlam*cosalp+(cosqK*Sdotn-cosqS)/sinqK*sinlam*sinalp;
      beta=ArcT(betadown,betaup);
      gam2=2.*(gim+beta);
      cos2gam=cos(gam2);
      sin2gam=sin(gam2);
    }

    for(n=1;n<nmodes+1;n++) {
      fn=n*nu+gimdotvec[i]/pi;
      Doppler=2.*pi*fn*AUsec*sinqS*cosorbphs;
      if (mich)
              nPhi=n*Phi+Doppler;
      else
              nPhi=n*Phi;

      if (TruncEcc) {
        nphiP = nPhi + gam2;
        Aplus = -0.5*n*Xp[n]*( Ap1*cos(nphiP) + Ap2*sin(nphiP) );
        Acros = -n*Xp[n]*( Ac1*cos(nphiP) + Ac2*sin(nphiP) );
      } else {
        ne=n*e;
        Jm2=Jn(n-2,ne);
        Jm1=Jn(n-1,ne);
        Jm0=Jn(n,ne);
        Jp1=Jn(n+1,ne);
        Jp2=Jn(n+2,ne);
        a=-n*(Jm2-2.*e*Jm1+2./n*Jm0+2.*e*Jp1-Jp2)*cos(nPhi);
        b=-n*sqrt(1-e*e)*(Jm2-2.*Jm0+Jp2)*sin(nPhi);
        c=2.*Jm0*cos(nPhi);

        Aplus=-(1.+Ldotn2)*(a*cos2gam-b*sin2gam)+c*(1-Ldotn2);
        Acros=2.*Ldotn*(b*cos2gam+a*sin2gam);
      }
      hnI=prefact*(FplusI *Aplus+FcrosI *Acros);
      hnII=prefact*(FplusII*Aplus+FcrosII*Acros);

      if (noiseweight)
        nfact=1./sqrt(sh(fn));
      else
        nfact=1.;

      hI[i]=hI[i]+hnI*nfact;
      hII[i]=hII[i]+hnII*nfact;
    }
    hI[i]*=Amp;
    hII[i]*=Amp;
  }

  for (i=nwave+1;i<=vlength;i++) {
    hI[i]=0.;
    hII[i]=0.;
  }

  PhiT=0;

  for(i=i0-1;i>=0;i--) {

    /* If (WCN.eq.0) then
       call NoiseInst (nuvec(i),gimdot0,nmodes,invsqrtS)
       else
       call NoiseFull (nuvec(i),gimdot0,nmodes,invsqrtS)
       endif*/

    hI[i]=0.;
    hII[i]=0.;

    //     t is actual time (sec) at step i
    t=tstart+timestep*(double)i;

    e=evec[i];
    nu=nuvec[i];
    Phi=Phivec[i];
    gim=gimvec[i];
    alp=alpvec[i];
    cosalp=cos(alp);
    sinalp=sin(alp);

    if (TruncEcc) {
      cos2alp=2.*cosalp*cosalp-1.;
      sin2alp=2.*cosalp*sinalp;
    } else {
       cosqL=cosqK*coslam+sinqK*sinlam*cosalp;
       sinqL=sqrt(1-cosqL*cosqL);
       BB=sinqK*cosphiK*coslam+sinphiK*sinlam*sinalp-cosqK*cosphiK*sinlam*cosalp;
       CC=sinqK*sinphiK*coslam-cosphiK*sinlam*sinalp-cosqK*sinphiK*sinlam*cosalp;
       phiL=ArcT(BB,CC);
       Ldotn=cosqL*cosqS+sinqL*sinqS*cos(phiL-phiS);
       Ldotn2=Ldotn*Ldotn;
    }
    if (mich) {
      orbphs=2.*pi*t/year+phi0;
      cosorbphs=cos(orbphs-phiS);
      sinorbphs=sin(orbphs-phiS);
      cosq=.5*cosqS-halfsqrt3*sinqS*cosorbphs;
      phiw=orbphs+phiw0+ArcT(sinqS*sinorbphs,halfsqrt3*cosqS+.5*sinqS*cosorbphs);
      psiup=.5*cosqK-halfsqrt3*sinqK*cos(orbphs-phiK)-cosq*(cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS));
      psidown=.5*sinqK*sinqS*sin(phiK-phiS)-halfsqrt3*cos(orbphs)*(cosqK*sinqS*sin(phiS)-cosqS*sinqK*sin(phiK))-halfsqrt3*sin(orbphs)*(cosqS*sinqK*cos(phiK)-cosqK*sinqS*cos(phiS));
      psi=ArcT(psidown,psiup);
      cosq1=.5*(1+cosq*cosq);
      cos2phi=cos(2.*phiw);
      sin2phi=sin(2.*phiw);
      cos2psi=cos(2.*psi);
      sin2psi=sin(2.*psi);

      FplusI=cosq1*cos2phi*cos2psi-cosq*sin2phi*sin2psi;
      FcrosI=cosq1*cos2phi*sin2psi+cosq*sin2phi*cos2psi;
      FplusII=cosq1*sin2phi*cos2psi+cosq*cos2phi*sin2psi;
      FcrosII=cosq1*sin2phi*sin2psi-cosq*cos2phi*cos2psi;
    } else {
      FplusI=1.;
      FcrosI=0.;
      FplusII=0.;
      FcrosII=1.;
    }

    Amp=pow(2.*pi*nu*M,2./3.)*zeta;

    if (TruncEcc) {
      gam2=2.*gim;
      cos2gam=cos(gam2);
      sin2gam=sin(gam2);

      Ap1 = Bp1c1*cosalp + Bp1c2*cos2alp + Bp1s1*sinalp+Bp1s2*sin2alp + Bp1cn;
      Ap2 = Bp2c1*cosalp + Bp2c2*cos2alp + Bp2s1*sinalp + Bp2s2*sin2alp + Bp2cn;
      Ac1 = Bc1c1*cosalp + Bc1c2*cos2alp + Bc1s1*sinalp+ Bc1s2*sin2alp + Bc1cn;
      Ac2 = Bc2c1*cosalp + Bc2c2*cos2alp + Bc2s1*sinalp+Bc2s2*sin2alp + Bc2cn;

      e2=e*e;
      Xp[1] = (-3. + 13./8.*e2)*e;
      Xp[2] = 2. - e2*(5. - 23./8.*e2);
      Xp[3] = e*(3. -e2*( 57./8. - 321./64.*e2));
      Xp[4] = e2*(4. - e2*(10. - 101./12.*e2));
      Xp[5] = e*e2*(125./24. - e2*(5375./384. - e2*42125./3072.));
    } else {
       Sdotn=cosqK*cosqS+sinqK*sinqS*cos(phiK-phiS);
       betaup=-Sdotn+coslam*Ldotn;
       betadown=sinqS*sin(phiK-phiS)*sinlam*cosalp+(cosqK*Sdotn-cosqS)/sinqK*sinlam*sinalp;
       beta=ArcT(betadown,betaup);
       gam2=2.*(gim+beta);
       cos2gam=cos(gam2);
       sin2gam=sin(gam2);
    }

    for(n=1;n<nmodes+1;n++) {
      fn=n*nu+gimdotvec[i]/pi;
      Doppler=2.*pi*fn*AUsec*sinqS*cosorbphs;
      if (mich)
              nPhi=n*Phi+Doppler;
      else
              nPhi=n*Phi;

      if (TruncEcc) {
        nphiP = nPhi + gam2;
        Aplus = -0.5*n*Xp[n]*( Ap1*cos(nphiP) + Ap2*sin(nphiP) );
        Acros = -n*Xp[n]*( Ac1*cos(nphiP) + Ac2*sin(nphiP) );
      } else {
         ne=n*e;
         Jm2=Jn(n-2,ne);
         Jm1=Jn(n-1,ne);
         Jm0=Jn(n,ne);
         Jp1=Jn(n+1,ne);
         Jp2=Jn(n+2,ne);
         a=-n*(Jm2-2.*e*Jm1+2./n*Jm0+2.*e*Jp1-Jp2)*cos(nPhi);
         b=-n*sqrt(1-e*e)*(Jm2-2.*Jm0+Jp2)*sin(nPhi);
         c=2.*Jm0*cos(nPhi);

         Aplus=-(1.+Ldotn2)*(a*cos2gam-b*sin2gam)+c*(1-Ldotn2);
         Acros=2.*Ldotn*(b*cos2gam+a*sin2gam);
      }

      hnI=prefact*(FplusI *Aplus+FcrosI *Acros);
      hnII=prefact*(FplusII*Aplus+FcrosII*Acros);

      if (noiseweight)
        nfact=1./sqrt(sh(fn));
      else
        nfact=1.;

      hI[i]=hI[i]+nfact*hnI;
      hII[i]=hII[i]+nfact*hnII;
    }
    hI[i]*=Amp;
    hII[i]*=Amp;
  }
  free(evec);
  free(nuvec);
  free(alpvec);
  free(Phivec);
  free(gimvec);
  free(gimdotvec);
  return;
}

void PNevolution(int vlength, double timestep, double *par, double nu0, double *gimdotvec, double *e, double *nu, double *Phi,
                 double *gim, double *alp, int *nwave) {
  int i,i0;
  double tend;
  double gimdot;
  double edot,nudot,Phidot,alpdot;
  double edotp,nudotp,Phidotp,gimdotp,alpdotp;
  double edotm,nudotm,Phidotm,gimdotm,alpdotm;
  double t0,mu,M,e0,Phi0,qS,phiS,gim0;
  double lam,alp0,S,qK,phiK;
  double Z,Y,hour,e2;
  double cosqS,sinqS,coslam,sinlam,cosqK,sinqK;
  double cosalp0,sinalp0;
  double SdotN,L0dotN,NcrossSdotL0,kappa0;

  hour=3600.;

  t0=0.;
  mu=par[logmu];
  M= par[logM];
  e0=par[ez];
  gim0=par[gamz];
  Phi0=par[phiz];
  cosqS=cos(par[colati]);
  phiS=par[longi];
  coslam=cos(par[Lambda]);
  alp0=par[alpz];
  S=par[SMBHSpin];
  cosqK=cos(par[thetaSpin]);
  phiK=par[phiSpin];

  sinqS=sin(par[colati]);
  sinlam=sin(par[Lambda]);
  sinqK=sin(par[thetaSpin]);
  cosalp0=cos(alp0);
  sinalp0=sin(alp0);

  /*     ----------------------------------------------
         HERE CALCULATE gam0 AS A FUNCTION OF gamtilde0

         SdotN=cosqK*cosqS+sinqK*sinqS*dcos(phiK-phiS)
         L0dotN=SdotN*coslam+(cosqS-SdotN*cosqK)*sinlam*cosalp0/sinqK
         p       +sinqS*sin(phiK-phiS)*sinlam*sinalp0
         NcrossSdotL0=sinqS*dsin(phiK-phiS)*sinlam*cosalp0
         p             -(cosqS-SdotN*cosqK)*sinlam*sinalp0/sinqK
         kappa0=ArcT(SdotN-coslam*L0dotN,NcrossSdotL0)
         gam0=gamtilde0+kappa0-pi/2
         write(*,*) gam0
         ---------------------------------------------- */

  i0=(int)(t0/timestep);

  e[i0]=e0;
  nu[i0]=nu0;
  Phi[i0]=Phi0;
  gim[i0]=gim0;
  alp[i0]=alp0;

  //     <<< EVOLVE FORWARD FROM t0 to tend >>>
  int ilast=vlength;
  bool plunged=false;

  for (i=i0;i<vlength;i++) {
    Z=pow(2.*pi*M*nu[i],1./3.);
    e2=e[i]*e[i];
    Y=1./(1.-e2);
    edotm=edot;
    nudotm=nudot;
    Phidotm=Phidot;
    gimdotm=gimdot;
    alpdotm=alpdot;
    edot=e[i]*mu/M/M*(-1./15.*pow(Y,3.5)*pow(Z,8.)*((304.+121.*e2)/Y+Z*Z*(70648.-231960.*e2-56101.*e2*e2)/56.)
                      +S*coslam*pow(Z,11.)*pow(Y,4.)*(8184.+10064.*e2+789.*e2*e2)/30.);
    nudot=96./(10.*pi)*mu/pow(M,3)*(pow(Z,11.)*pow(Y,4.5)*((96.+292.*e2+37.*e2*e2)/Y/96.
                                                           +Z*Z*(20368.-61464.*e2-163170.*e2*e2-13147.*e2*e2*e2)/5376.)
                                    -pow(Z,14.)*pow(Y,5.)*S*coslam*(1168.+9688.*e2+6286.*e2*e2 +195.*e2*e2*e2)/192.);
    Phidot=2.*pi*nu[i];
    alpdot=8.*pi*pi*nu[i]*nu[i]*S*M*pow(Y,1.5);
    gimdot=6.*pi*nu[i]*Z*Z*Y*(1.+.25*Z*Z*Y*(26.-15.*e2))-3.*coslam*alpdot;

    if (i == i0) {
      edotm=edot;
      nudotm=nudot;
      Phidotm=Phidot;
      gimdotm=gimdot;
      alpdotm=alpdot;
      gimdotm=gimdot;
    }
    gimdotvec[i]=gimdot;
    e[i+1]=e[i]+(1.5*edot-.5*edotm)*timestep;
    nu[i+1]=nu[i]+(1.5*nudot-.5*nudotm)*timestep;
    Phi[i+1]=Phi[i]+(1.5*Phidot-.5*Phidotm)*timestep;
    gim[i+1]=gim[i]+(1.5*gimdot-.5*gimdotm)*timestep;
    alp[i+1]=alp[i]+(1.5*alpdot-.5*alpdotm)*timestep;
    if (nu[i+1] > pow((1.-e[i+1]*e[i+1])/(2.*(3.+e[i+1])),1.5)/(2.*pi*M)) {
      plunged=true;
      ilast=i;
      i=vlength;
    }
  }
  gimdotvec[ilast]=gimdot;

  //     <<< EVOLVE BACKWARD FROM t0 to t=0 >>>

  for (i=i0;i>0;i--) {
    Z=pow(2.*pi*M*nu[i],1./3.);
    e2=e[i]*e[i];
    Y=1./(1.-e2);
    edotp=edot;
    nudotp=nudot;
    Phidotp=Phidot;
    gimdotp=gimdot;
    alpdotp=alpdot;

    edot=e[i]*mu/M/M*(-1./15.*pow(Y,3.5)*pow(Z,8.)*((304.+121.*e2)/Y+Z*Z*(70648.-231960.*e2-56101.*e2*e2)/56.)
                      +S*coslam*pow(Z,11.)*pow(Y,4.)*(8184.+10064.*e2+789.*e2*e2)/30.);
    nudot=96./(10.*pi)*mu/pow(M,3)*(pow(Z,11.)*pow(Y,4.5)*((96.+292.*e2+37.*e2*e2)/Y/96.
                                                           +Z*Z*(20368.-61464.*e2-163170.*e2*e2-13147.*e2*e2*e2)/5376.)
                                    -pow(Z,14.)*pow(Y,5.)*S*coslam*(1168.+9688.*e2+6286.*e2*e2 +195.*e2*e2*e2)/192.);
    Phidot=2.*pi*nu[i];
    alpdot=8.*pi*pi*nu[i]*nu[i]*S*M*pow(Y,1.5);
    gimdot=6.*pi*nu[i]*Z*Z*Y*(1.+.25*Z*Z*Y*(26.-15.*e2))-3.*coslam*alpdot;

    if (i == i0) {
      edotp=edot;
      nudotp=nudot;
      Phidotp=Phidot;
      gimdotp=gimdot;
      alpdotp=alpdot;
    }
    e[i-1]=e[i]-(1.5*edot-.5*edotp)*timestep;
    nu[i-1]=nu[i]-(1.5*nudot-.5*nudotp)*timestep;
    Phi[i-1]=Phi[i]-(1.5*Phidot-.5*Phidotp)*timestep;
    gim[i-1]=gim[i]-(1.5*gimdot-.5*gimdotp)*timestep;
    alp[i-1]=alp[i]-(1.5*alpdot-.5*alpdotp)*timestep;
    gimdotvec[i]=gimdot;
  }
  gimdotvec[0]=gimdot;

  *nwave=ilast;
  return;
}

/* Generate a Barack and Cutler waveform, assuming that the waveform is parameterised by parameters at the starting time, t=0. */

void GenBCWaveStart(TDIframework *fw, double *par,double *hI, double *hII)
{
  double deltat=fw->dataDeltaT;
  double vlength=fw->dataN;
  int i,n,nmodes,p,pp,m,k,iv;
  int dim,direction,comp,pointM,pointe0;
  int timestep1,nsteps1;
  double dt,t,spinres,dirres;

  double bigest,count,sqrtdet;
  double integral,StoN,eisco,nuisco,gimdot0;
  double delta,deltahalf,t0,q[NParams],element;
  double parvals[NParams];
  double normI,normII,norm,dtdays;
  double t0res,mures,Mres,e0res,gim0res,Phi0res;
  double tpl,lamres,alp0res,OmegaSres,Dres,zeta,invD,Gpsc;

  for (i=0;i<NParams;i++)
    parvals[i]=par[i];
 
  dt=deltat*((double)vlength);
  /* Set parameters for waveform. */
  //double nust=par[lognuz];
  //parvals[logmu]=par[logmu]*SOLARMASSINSEC;
  //parvals[logM]= par[logM]*SOLARMASSINSEC;

  //parvals[lognuz]=0.;
  //zeta=parvals[logmu]/(exp(parvals[logDl])*GPCINSEC);

  /* Set parameters for waveform. Logrithmic Values */
  double nust=exp(par[lognuz]);
  parvals[logmu]=exp(par[logmu])*SOLARMASSINSEC;
  parvals[logM]=exp(par[logM])*SOLARMASSINSEC;

  parvals[lognuz]=0.;
  zeta=parvals[logmu]/(exp(parvals[logDl])*GPCINSEC);
  
 /* -------------------------------------------------------------
     NUMBER OF MODES TO BE CONSIDERED (as a function of e0)
     ------------------------------------------------------------- */

  /*     This is an experimental result: See Peters & Mathews fig 3
         and Eq. (20), and see ModePower.nb */
  if (TruncEcc)
    nmodes=5;
  else {
    nmodes=(int)(30*parvals[ez]);
    if (parvals[ez] < 0.135) nmodes=4;
  }

  /* --------------------------------------------------------------
     CALCULATING WAVEFORM & DERIVATIVES
     -------------------------------------------------------------- */

  waveform(0.,parvals,nust,vlength,deltat,hI,hII,nmodes,zeta,0);
  return;
}


void generateresiduals(TDIframework *fw, double *parameter)
/*********************************************/
/* generates waveforms etc. and leaves the   */
/* (proposed) residuals in                   */
/* `fw->AresidualPro' and `fw->EresidualPro' */
/*********************************************/
{
  double ChiSquaredA, ChiSquaredE;
  double subtotalA, subtotalE;
  double absdiffA, absdiffE;
  double *plus, *cross;
  double *Aresponse, *Eresponse;
  fftw_complex *AresponseFT, *EresponseFT;
  long i;

int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* write plus- & cross waveform to `plus' & `cross': */
  plus  = (double*) malloc((fw->dataN+100) * sizeof(double));
  cross = (double*) malloc((fw->dataN+100) * sizeof(double));
  //setGWbarycenter(fw, parameter, plus, cross); 
  GenBCWaveStart(fw, parameter, plus, cross); 

  /* Fourier transform the A/E responses: */
  AresponseFT = (fftw_complex*) malloc(fw->FToutLength * sizeof(fftw_complex));
  FTexec(fw, plus, AresponseFT);
  free(plus);
  EresponseFT = (fftw_complex*) malloc(fw->FToutLength * sizeof(fftw_complex));
  FTexec(fw, cross, EresponseFT);
  free(cross);
  // derive the A & E residuals:
  for (i=0; i<fw->FToutLength; ++i){
    absdiffA = cabs(fw->AdataFT[i] - AresponseFT[i]);
    absdiffE = cabs(fw->EdataFT[i] - EresponseFT[i]);
    fw->AresidualPro[i] = absdiffA*absdiffA;
    fw->EresidualPro[i] = absdiffE*absdiffE;
    //  -->  these are now the sum of squared real & imag FFTW parts
  }
  free(AresponseFT);
  free(EresponseFT);
}



double signaltonoiseratio(TDIframework *fw, double *parameter)
/**************************/
/*  frequency-domain 
  */
/**************************/
{
  double ChiSquaredA, ChiSquaredE;
  double subtotalA, subtotalE, absval;
  double *plus, *cross;
  double *Aresponse, *Eresponse;
  fftw_complex *AresponseFT, *EresponseFT;
  long i, j, k;
  double loglikelihood;
  long root;
  double snrA, snrE;

  /* write plus- & cross waveform to `plus' & `cross': */
  plus  = (double*) malloc((fw->dataN +100) * sizeof(double));
  cross = (double*) malloc((fw->dataN +100) * sizeof(double));
  GenBCWaveStart(fw, parameter, plus, cross); 

  /* Fourier transform the A/E responses: */
  AresponseFT = (fftw_complex*) malloc(fw->FToutLength * sizeof(fftw_complex));
  FTexec(fw, plus, AresponseFT);
  free(plus);
  EresponseFT = (fftw_complex*) malloc(fw->FToutLength * sizeof(fftw_complex));
  FTexec(fw, cross, EresponseFT);
  free(cross);
  /* sum up the squared differences:                       */
  /* (via subtotals, in order to minimise rounding errors) */
  root = ceil(sqrt(((double) fw->FToutLength)));
  ChiSquaredA = ChiSquaredE = 0.0;
  for (i=0; i<root; ++i){
    subtotalA = subtotalE = 0.0;
    for (j=0; j<root; ++j){
      k = i*root + j + 1; // `+1' so very first bin (zero frequency) isn't counted.
      if (k >= fw->FToutLength) j=root; // ends this loop
      else {
        if ((fw->FTfreq[k]>freqBandLower) & (fw->FTfreq[k]<freqBandUpper)){ 
          absval = cabs(AresponseFT[k]);
          //subtotalA += exp(2.0*log(absval) - (fw->AnoisePSD[k]+1.386294));
          subtotalA += exp(2.0*log(absval) - fw->AnoisePSD[k]);
          absval = cabs(EresponseFT[k]);
          //subtotalE += exp(2.0*log(absval) - (fw->EnoisePSD[k]+1.386294));
          subtotalE += exp(2.0*log(absval) - fw->EnoisePSD[k]);
        }
      }
    }
    ChiSquaredA += subtotalA;
    ChiSquaredE += subtotalE;
  }
  free(AresponseFT);
  free(EresponseFT);
  snrA = 2.0*sqrt(ChiSquaredA/(double)(scalehigh-scalelow));
  snrE = 2.0*sqrt(ChiSquaredE/(double)(scalehigh-scalelow));
  printf(" : SNR_A=%.3f, SNR_E=%.3f\n",snrA,snrE);
  return sqrt(snrA*snrA + snrE*snrE);
}


double tukey(int j, int N, double r)
/* Tukey window... for r=0 equal to rectangular window, for r=1 equal to Hann window. */
/* ( 0 < r < 1 denotes the fraction of the window in which it behaves sinusoidal)     */
/* j = 0, ..., N-1                                                                    */
{
  double win = 1.0;
  if (((double)j)>(((double)N)/2.0)) j = N-j;
  if (((double)j) < (r*(((double)N)/2.0)))
    win = 0.5*(1.0-cos(((2.0*pi)/r)*(((double)j)/((double)N))));
  return win;
}

double logprior(double *parameter, TDIframework *fw)
/***********************************************************/
/* log- prior density for a given parameter vector.        */
/* Unnormalised, i.e. up to a normalising constant factor. */
/* Returns `-HUGE_VAL' for zero density.                   */
/***********************************************************/
{
  double optiampli = log(0.5) + (5.0/6.0)*log(prior_reference_mt);
  double ampli90 = optiampli - log(prior_dist_90);
  double ampli10 = optiampli - log(prior_dist_10);
  double a = (ampli10+ampli90)/2.0;
  double b = (ampli10-ampli90)/(-2.0*log(0.1/0.9));
  double ampli;
  //-----------------------------------------------

  double lprior = 0.0;
  double sigmasq2spec = 2.0*log(fw->dataDeltaT) + log(fw->dataN);
  double logsigmasq;
  double exponent  = -(PSDpriorDF/2.0 + 1.0);
  double log2 = log(2.0);
  long i;

   /* lognuz */
  lprior += ((parameter[lognuz]< log(0.25))) ? parameter[lognuz] : -HUGE_VAL;

  /* logmu and logM */
  lprior += ((parameter[logmu]>= log(5.5)) && (parameter[logmu] <= log(10.5))) ? parameter[logmu] : -HUGE_VAL;
  lprior += ((parameter[logM] >= log(4.75e5)) && (parameter[logM] <= log(1.05e7))) ? parameter[logM] : -HUGE_VAL;
  
  /* prior proportional to sin(latitude): */
  lprior += ((parameter[colati]>0.0) && (parameter[colati]<pi)) ? log(sin(parameter[colati])) : -HUGE_VAL;
  
  /* uniform prior for longitude: */
  lprior += ((parameter[longi]>=0.0) && (parameter[longi]<twopi)) ? 0.0 : -HUGE_VAL;
  
  /* uniform prior for phase: */
  lprior += ((parameter[gamz]>=0.0) && (parameter[gamz]<twopi)) ? 0.0 : -HUGE_VAL;  
  lprior += ((parameter[phiz]>=0.0) && (parameter[phiz]<twopi)) ? 0.0 : -HUGE_VAL;  
  lprior += ((parameter[alpz]>=0.0) && (parameter[alpz]<twopi)) ? 0.0 : -HUGE_VAL;  
  
  /* Spin */
  lprior += ((parameter[thetaSpin]>=0.0) && (parameter[thetaSpin]<pi)) ? log(sin(parameter[thetaSpin])) : -HUGE_VAL; 
  lprior += ((parameter[phiSpin]>=0.0) && (parameter[phiSpin]<twopi)) ? 0.0 : -HUGE_VAL; 

  lprior += ((parameter[Lambda]>=0.0) && (parameter[Lambda]< pi)) ? log(sin(parameter[Lambda])) : -HUGE_VAL;    
  lprior += ((parameter[SMBHSpin]>=0.5) && (parameter[SMBHSpin]<= 0.7)) ? 0.0 : -HUGE_VAL;
  
  /* Eccentricity */
  lprior += ((parameter[ez] >= 0.15) && (parameter[ez] <= 0.30)) ? 0.0: -HUGE_VAL;
  
  /* log Distance GPC */ 
  lprior += (parameter[logDl]<=2.0) ? 2.0*parameter[logDl] : -HUGE_VAL;
  
  //if (lprior > -HUGE_VAL){
    //for (i=0; i<fw->FToutLength; ++i){
      //if ((fw->FTfreq[i]>freqBandLower) & (fw->FTfreq[i]<freqBandUpper)){ 
        ///* transform the PSD to `sigma' scale: */
        //logsigmasq = fw->AnoisePSD[i] - sigmasq2spec;
        ///* prior density: */
        //lprior += exponent*logsigmasq - exp(log(PSDpriorDF*fw->AsigmaPrior[i])-log2-logsigmasq);
      //}
    //} 
    //if (! AEcommonSpectrum){ // do same for `E' spectrum:
      //for (i=0; i<fw->FToutLength; ++i){
        //if ((fw->FTfreq[i]>freqBandLower) & (fw->FTfreq[i]<freqBandUpper)){ 
          //logsigmasq = fw->EnoisePSD[i] - sigmasq2spec;
          //lprior += exponent*logsigmasq - exp(log(PSDpriorDF*fw->EsigmaPrior[i])-log2-logsigmasq);
	//}
      //} 
    //}
  //}
  
  return lprior;
}

void propose(double *state, double *new, double temperature)
/* generate a proposal `new'               */
/* dependent on the current state `state'. */
/* Proposals are generated using a Student-t distribution. */
{
  double propscale = adj*sqrt(temperature);
  double posvec[3], hingevec[3];
  double sinlati;
  double dummy[14];
  double step[14];
  double studentDF = 3.0;
  int i;

  // generate multivariate-normal random vector: 
  genmn(MVNpar, step, dummy);
  // turn into student-T distributed RV:
  if (studentDF > 0.0)
    for (i=0; i<14; ++i)
      step[i] *= propscale * sqrt(studentDF/genchi(studentDF));
  
  //for (i=0; i<14; ++i)
      //printf("step[%d] %.10f\n",i, step[i]);
  
  /* initialise `new': */
  for (i=0;i<14;i++)
	new[i]=state[i]; 

    new[lognuz]      = state[lognuz] + step[lognuz];
    new[logmu]       = state[logmu] + step[logmu];
    new[logM]        = state[logM] + step[logM];
    new[ez]          = state[ez] + step[ez];
    new[gamz]        = state[gamz] + step[gamz];
    new[phiz]        = state[phiz] + step[phiz];
    new[colati]      = state[colati] + step[colati];
    new[longi]       = state[longi] + step[longi];
    new[Lambda]      = state[Lambda] + step[Lambda];
    new[alpz]        = state[alpz] + step[alpz];
    new[SMBHSpin]    = state[SMBHSpin] + step[SMBHSpin];
    new[thetaSpin]   = state[thetaSpin] + step[thetaSpin];
    new[phiSpin]     = state[phiSpin] + step[phiSpin];
    new[logDl]       = state[logDl] + step[logDl];

	
  ///* adjust for out-of-domain proposals: */
  while (new[gamz] >= twopi)  new[gamz] -= twopi;
  while (new[gamz] < 0.0)  new[gamz] += twopi;
  while (new[phiz] >= twopi)  new[phiz] -= twopi;
  while (new[phiz] < 0.0)  new[phiz] += twopi;
  while (new[alpz] >= twopi)  new[alpz] -= twopi;
  while (new[alpz] < 0.0)  new[alpz] += twopi;
  if (new[colati]  >= pi) { 
    new[longi] += pi;
    new[colati] = twopi - new[colati];
  }
  if (new[colati]  <= 0.) { 
    new[longi] += pi;
    new[colati] = -new[colati];
  }
  if (new[thetaSpin]  >= pi) { 
    new[phiSpin] += pi;
    new[thetaSpin] = twopi - new[thetaSpin];
  }
  if (new[thetaSpin]  <= 0.) { 
    new[phiSpin] += pi;
    new[thetaSpin] = -new[thetaSpin];
  }
  if (new[Lambda]  >= pi) { 
    new[alpz] += pi;
    new[Lambda] = twopi - new[Lambda];
  }
  if (new[Lambda]  <= 0.) { 
    new[alpz] += pi;
    new[Lambda] = -new[Lambda];
  }
  while (new[longi] >= twopi) new[longi] -= twopi;
  while (new[longi] <  0.0)   new[longi] += twopi;
  while (new[phiSpin] >= twopi) new[phiSpin] -= twopi;
  while (new[phiSpin] <  0.0)   new[phiSpin] += twopi;
}

void datamatrix(int i, double state[14], double temperature, double datamat[dataend][14])
{   
	int a;
	if((i>0) & (i<=dataend))
	{
		for(a=0; a<14; a++)
		{
		datamat[i][a] = state[a];
	    }
	}
}

void covmat(int i, double data[dataend][14], int datasize, double temperature)
{
	int b, c, d;
	double meancol[14], covmatrix[14][14];
	double zeromean[14];
    double dummycov[196];
	if(i==dataend)
	{	
		for(c=0; c<14; c++)
		{
			meancol[c] = 0.0;
			for(b=datastart+1; b<=dataend; b++)
			{
				 meancol[c] += data[b][c];     // Computes the means of data columns
			}
			meancol[c] /= datasize; 
		}
			
		for (b =datastart+1; b <=dataend; b++)
		{
			for (c = 0; c < 14; c++)
			{
				data[b][c] -= meancol[c];     // Centralize the data columns by mean
			}
		}

		for (d = 0;  d < 14; d++)
		{
			for (c = 0; c < 14; c++)
			{
				covmatrix[d][c] = 0.0;
				for (b = datastart+1; b <=dataend; b++)
				{
					covmatrix[d][c] += data[b][d] * data[b][c];
				}
					covmatrix[d][c]/= (datasize-1);
					covmatrix[d][c]/= temperature;
					covmatrix[d][c]/=adj;

			}
		}

		for(d=0; d<14; d++)
		{
			for(c=0; c<14; c++)
			{
				PropCov[d][c] = covmatrix[d][c];	
			}
		}
		
		for (d=0; d<14; ++d)
		{
			zeromean[d] = 0.0;
			for (c=0; c<14; ++c)
			{
				dummycov[d*14+c] = PropCov[d][c];
			}
		}
		
		if(temperature == 1)		
		   {
			for(b=0; b<14; b++)
				{
					for(d=0; d<14; d++)
					{
						printf("  %.3le", PropCov[b][d]);
					}
					printf("\n");
				}
	   		}
 setgmn(zeromean, dummycov, 14, MVNpar); 
 	}
}

int Kj(long j, int N)
{ 
	int k;
	if ((j==0) || (((N%2)==0) && (j==N/2))){
		k=0;
		} else {
		k=1;
		}
		return k;
}

void drawspectrum(TDIframework *fw, double temperature)
/* draws the CONDITIONAL POSTERIOR SPECTRUM,   */
/* conditional on the current set of residuals */
/* given in the vector `fw->AresidualAcc' and  */
/* stores it in  `fw->AnoisePSD'  (same for    */
/* `E' channel).                               */
{
  long i,j,k;
  double amplitudessq;
  double logchisq;
  double sigmasq2spec = 2.0*log(fw->dataDeltaT) + log(fw->dataN);
  double DFT2AB = (4.0/pow((double)fw->dataN,2.0));
  double logsigma2;
  double df;
  double scale;
  //  draw the CONDITIONAL POSTERIOR SPECTRUM:
  if (! AEcommonSpectrum){ // individual A/E spectra:
    df = PSDpriorDF + 2.0;
    for (j=0; j<fw->FToutLength; ++j){
      // derive the amplitudes' (a,b) sum-of-squares:
      //amplitudessq = (4.0/pow((double)fw->dataN,2.0)) * fw->AresidualAcc[j]; //  = a[j]^2 + b[j]^2
      // derive a (tempered) conditional posterior draw of sigma^2:
      //logchisq = log(genchi(df));
      logsigma2 = log(fw->Asigma[j]);// + amplitudessq);//-logchisq;
      // derive the corresponding (conditional posterior) spectrum:
      fw->AnoisePSD[j] = logsigma2 + sigmasq2spec;
      // same for `E' channel...:
      //amplitudessq = (4.0/pow((double)fw->dataN,2.0)) * fw->EresidualAcc[j]; //  = a[j]^2 + b[j]^2
      //logchisq = log(genchi(df));
      logsigma2 = log(fw->Esigma[j]);// + amplitudessq);//-logchisq;
      fw->EnoisePSD[j] = logsigma2 + sigmasq2spec;
    }
  }
  else{ // common A/E spectrum:
    df = PSDpriorDF + 4.0;
    for (j=0; j<fw->FToutLength; ++j){
      // derive the amplitudes' (a,b) sum-of-squares:
      //amplitudessq  = fw->AresidualAcc[j];
      //amplitudessq += fw->EresidualAcc[j];
      //amplitudessq *= DFT2AB; // = a_A[j]^2+b_A[j]^2+a_E[j]^2+b_E[j]^2
      // derive a (tempered) conditional posterior draw of sigma^2:
      //logchisq = log(genchi(df));
      logsigma2 = log(fw->Asigma[j]);// + amplitudessq);//-logchisq;
      // derive the corresponding (conditional posterior) spectrum:
      fw->AnoisePSD[j] = logsigma2 + sigmasq2spec;
      // same for `E' channel...:
      fw->EnoisePSD[j] = fw->AnoisePSD[j];
    }   
  }
}

//double likelihood(TDIframework *fw, double *AresiVec, double *EresiVec)//, double lprior, double *newlogposterior)
///* The corresponding updated log-posterior     */
///* value is returned in  `newlogposterior'.    */
//{
  //long i,j,k;
  //long root;
  //double ChiSquaredA, ChiSquaredE;
  //double compA1, compA2, compE1, compE2;
  //double subtotalA, subtotalE;
  //double loglikelihood;
  ////double logcoef = log(2.0/(fw->deltaFT) * pow(fw->dataDeltaT,2.0));
  //double logcoef = log(2.0) - log(fw->deltaFT) + 2.0*log(fw->dataDeltaT);
  ////double logcoef2 = log(4.0) + 2.0*log(fw->dataDeltaT);
  //double logcoef2 = log(2.0) + 2.0*log(fw->dataDeltaT) - log(fw->dataN);
  //double coef3 = (fw->dataDeltaT)/((fw->dataN)*PSDpriorDF);
  //root = ceil(sqrt(((double) fw->FToutLength)));
  //ChiSquaredA = ChiSquaredE = 0.0;
  //compA1 = compA2 = compE1 = compE2 = 0.0;
  //for (i=0; i<root; ++i){
	//subtotalA = subtotalE = 0.0;
     //for (j=0; j<root; ++j){
      //k = i*root + j + 1; // `+1' so very first bin (zero frequency) isn't counted.
      //if (k >= fw->FToutLength) j=root; // ends this loop
      //else {
        //if ((fw->FTfreq[k]>freqBandLower) & (fw->FTfreq[k]<freqBandUpper)){ 
               
                //compA1 += exp(log(AresiVec[k]) - fw->AnoisePSD[k] - log((double)(scalehigh-scalelow)));
                //compA2 += fw->AnoisePSD[k];
                //compE1 += exp(log(EresiVec[k]) - fw->EnoisePSD[k] - log((double)(scalehigh-scalelow)));
                //compE2 += fw->EnoisePSD[k];
                ////subtotalA = (fw->FToutLength) * log(compA1) + compA2;
                ////subtotalE = (fw->FToutLength) * log(compE1) + compE2;
	//}
      //}
	  //}
    //ChiSquaredA = ((double)(scalehigh-scalelow)) * log(compA1) + compA2;
    //ChiSquaredE = ((double)(scalehigh-scalelow)) * log(compE1) + compE2;
  //}

  //loglikelihood = -(ChiSquaredA + ChiSquaredE);
  //return(loglikelihood);
//}

double likelihood(TDIframework *fw, double *AresiVec, double *EresiVec)//, double lprior, double *newlogposterior)
/* The corresponding updated log-posterior     */
/* value is returned in  `newlogposterior'.    */
{
  long i,j,k;
  long root;
  double ChiSquaredA, ChiSquaredE;
  double subtotalA, subtotalE;
  double loglikelihood;
  //double logcoef = log(2.0/(fw->deltaFT) * pow(fw->dataDeltaT,2.0));
  double logcoef = log(2.0) - log(fw->deltaFT) + 2.0*log(fw->dataDeltaT);
  //double logcoef2 = log(4.0) + 2.0*log(fw->dataDeltaT);
  double logcoef2 = log(2.0) + 2.0*log(fw->dataDeltaT) - log((double)(scalehigh-scalelow));
  root = ceil(sqrt(((double) fw->FToutLength)));
  ChiSquaredA = ChiSquaredE = 0.0;
  for (i=0; i<root; ++i){
    subtotalA = subtotalE = 0.0;
    for (j=0; j<root; ++j){
      k = i*root + j + 1; // `+1' so very first bin (zero frequency) isn't counted.
      if (k >= fw->FToutLength) j=root; // ends this loop
      else {
        if ((fw->FTfreq[k]>freqBandLower) & (fw->FTfreq[k]<freqBandUpper)){ 
          subtotalA += exp(logcoef2 + log(AresiVec[k]) - fw->AnoisePSD[k]);
          subtotalE += exp(logcoef2 + log(EresiVec[k]) - fw->EnoisePSD[k]);
	}
      }
    }
    ChiSquaredA += subtotalA;
    ChiSquaredE += subtotalE;
  }
  //loglikelihood = -2.0/(fw->deltaFT) * pow(fw->dataDeltaT,2.0) * (ChiSquaredA + ChiSquaredE);
  loglikelihood = -(ChiSquaredA + ChiSquaredE);
  return(loglikelihood);
}


void metropolis(TDIframework *fw, double startpar[14], long iterations, 
                int MPIsize, int MPIrank)
{
  char completefilename[200]; 
  FILE *output = NULL;
  double state[14], prop[14];
  double lprior, lpriorProp;
  double loglh, loglhProp, logalpha, acceptRate;
  double loglikeli;
  int accept;
  long i, j, k, m;
  double largest;
  time_t starttime, lasttime, now;

  double *tempvec=NULL;
  int OwnTempIndex, prevTemp;
  struct tm *ltime;
  double *LLHVec=NULL;
  int *TempIndVec=NULL;
  long *Nproposed=NULL;
  long *Naccepted=NULL;
  long acceptCount=0;
  int swap1, swap2;
  int thinout=10;// from 100 to 1 by waqas, it writes output lines with what difference?
  double sigmasq2spec = 2.0*log(fw->dataDeltaT) + log(fw->dataN);
  double DFT2AB = (4.0/pow((double)fw->dataN,2.0));
  long SpectrumUpdateSkip = 10;
  long SpectrumLogSkip = 5000;
  double logspectrum;
  double *logSpectrumMean=NULL;
  //double *logSpectrumVar=NULL;
  double *residualSSQ=NULL;
  MPI_Status mpistat;
  long whichTempOne;
  long n;
  double weight1, weight2, weight3, diff;
  double snr;

/* Covariance Stuff*/
double datamat[dataend+10][14];
int b, c, d;
double dummycov[196];

  time(&starttime);
  // copy starting parameter to `state':
  for (j=0; j<14; ++j) state[j] = startpar[j];
  generateresiduals(fw, startpar);
  //intf(" | #%d: `generateresiduals()' done.\n",MPIrank);
  // this state is `accepted' per definitionem!
  // --> update residuals:
  for (j=0; j<fw->FToutLength; ++j){
    fw->AresidualAcc[j] = fw->AresidualPro[j];
    fw->EresidualAcc[j] = fw->EresidualPro[j];
  }
  // now draw the conditional spectrum
  drawspectrum(fw, 1.0);
  lprior = logprior(startpar, fw);
  //intf(" | #%d: `logprior()' done.\n",MPIrank);
  // and compute (conditional) likelihood:
  loglh = likelihood(fw, fw->AresidualAcc, fw->EresidualAcc);
  // define temperature ladder:
  tempvec = (double*) malloc(sizeof(double)*MPIsize);
  //for (j=0; j<MPIsize; ++j) tempvec[j] = pow(1.9, (double)j);
  for (j=0; j<MPIsize; ++j) tempvec[j] = pow(1.1, (double)j);
  residualSSQ = (double*) malloc(sizeof(double)*fw->FToutLength);

  // initialise stuff specific for process number zero:
  if (MPIrank == 0){
    // start the log file:
    sprintf(completefilename, "%s/%s-log.txt", outputpath, outputfile);
    output = fopen(completefilename,"w");
    ltime = localtime(&starttime);
    fprintf(output, "MCMC started: %02i.%02i.%04i  %02i:%02i:%02i\n",
            ltime->tm_mday, ltime->tm_mon+1, ltime->tm_year+1900, 
            ltime->tm_hour, ltime->tm_min, ltime->tm_sec);
    fprintf(output, "chains: %d,  temperature factor: %.4f\n", MPIsize, tempvec[1]);
    fprintf(output, "iterations seconds");
    for (j=0; j<(MPIsize-1); ++j) fprintf(output, " P%d", j+1);
    for (j=0; j<(MPIsize-1); ++j) fprintf(output, " A%d", j+1);
    fprintf(output, "\n0 0");
    for (j=0; j<(MPIsize-1); ++j) fprintf(output, " 0 0", j+1);
    fprintf(output, "\n");
    fclose(output);
    // start the spectrum file, 1st column is the "iteration" column,
    // first line contains frequencies for each frequency bin,
    // and second line contains the (log-) prior spectrum:
    sprintf(completefilename, "%s/%s-spectrum.txt", outputpath, outputfile);
    output = fopen(completefilename,"w");
    fprintf(output, "0 0");
    for (j=1; j<(fw->FToutLength-1); ++j)
      fprintf(output, " %.9f", fw->FTfreq[j]);
    fprintf(output, "\n");
    fprintf(output, "0 0");
    for (j=1; j<(fw->FToutLength-1); ++j)
      fprintf(output, " %.6e", log(fw->Asigma[j])+sigmasq2spec);
    fprintf(output, "\n");
    fclose(output);
    logSpectrumMean = (double*) malloc(sizeof(double)*fw->FToutLength);
    //logSpectrumVar  = (double*) malloc(sizeof(double)*fw->FToutLength);
    for (j=0; j<fw->FToutLength; ++j){
      logSpectrumMean[j] = 0.0;
      //logSpectrumVar[j]  = 0.0;
    }
    // `LLHVec' is used to gather all chains' loglikelihood values:
    LLHVec = (double*) malloc(sizeof(double)*MPIsize);
    // `TempIndVec' is used to manage all chains' temperature indices:
    TempIndVec = (int*) malloc(sizeof(int)*MPIsize);
    // `Nproposed' is used to log number of swap proposals for all pairs:
    Nproposed = (long*) malloc(sizeof(long)*(MPIsize-1));
    // `Naccepted' is used to log number of accepted swaps for all pairs:
    Naccepted = (long*) malloc(sizeof(long)*(MPIsize-1));
    for (j=0; j<(MPIsize-1); ++j) {
      Nproposed[j] = 0;
      Naccepted[j] = 0;
    }
    for (j=0; j<MPIsize; ++j) TempIndVec[j] = j;
    printf(" | temperature ladder: ");
    for (j=0; j<MPIsize; ++j) printf(" %.3f",tempvec[j]);
    printf("\n");

  }
  // determine temperature-indices for all processes (simply sort by likelihoods);
  // first gather likelihood values in process #0:
  MPI_Gather(&loglh, 1, MPI_DOUBLE,
             LLHVec, 1, MPI_DOUBLE,
             0, MPI_COMM_WORLD);
  // process #0 does the sorting:
  if (MPIrank == 0){
    for (j=0; j<MPIsize; ++j)
      TempIndVec[j] = -1;
    for (j=0; j<MPIsize; ++j){
      largest = -HUGE_VAL;
      for (k=0; k<MPIsize; ++k){
        if ((TempIndVec[k] == -1) && (LLHVec[k] > largest)) {
          m = k;
          largest = LLHVec[k];
        }
      }
      TempIndVec[m] = j;
    }
  }
  // distribute the temperature assignments to all processes:
  MPI_Scatter(TempIndVec, 1, MPI_INT,
              &OwnTempIndex, 1, MPI_INT,
              0, MPI_COMM_WORLD);   
  snr = signaltonoiseratio(fw, startpar);
  // write first 2 lines to (each) file:
  sprintf(completefilename, "%s/%s-C%d.txt", outputpath, outputfile, OwnTempIndex+1);
  output = fopen(completefilename,"a");
  
  fprintf(output, "iteration lognuz logmu logM ez gamz phiz colati longi Lambda alpz SMBHSpin thetaSpin phiSpin logDl logposterior loglikeli accept SNR\n");
  fprintf(output, "%d %.12f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.4f %.4f %d %.4f\n",
          0, startpar[lognuz], startpar[logmu], startpar[logM], startpar[ez], startpar[gamz], startpar[phiz], 
          startpar[colati], startpar[longi], startpar[Lambda], startpar[alpz], startpar[SMBHSpin], 
          startpar[thetaSpin], startpar[phiSpin], startpar[logDl],lprior+loglh, loglh, acceptCount, snr);
          fclose(output);
  
  // start Metropolis-sampler:
  for (i=1; i<=iterations; ++i){
	  
	 //printf("IN =%i\n", i);
	  //printf("PN =%d\t", OwnTempIndex+1);
    /* ==== START OF ITERATION ==== */
    // --->  firstly, the `regular' Metropolis-step:
    // generate proposal:
    propose(state, prop, tempvec[OwnTempIndex]);
    // compute prior & posterior:
    lpriorProp  = logprior(prop, fw);
    if (lpriorProp > -HUGE_VAL){
      generateresiduals(fw, prop);
      loglhProp = likelihood(fw, fw->AresidualPro, fw->EresidualPro);
    }
    else loglhProp = -HUGE_VAL;
    // determine -tempered- (log-) acceptance probability alpha:
    logalpha = (1.0/tempvec[OwnTempIndex])*(loglhProp-loglh);
    accept = ((logalpha >= 0.0) || (log(genunf(0.0,1.0)) < logalpha));
    // if accepted, update state &c.:
   // if (accept){
      for (j=0; j<14; ++j) state[j] = prop[j];      // (update state)
      lprior = lpriorProp;                         // (update prior...
      loglh = loglhProp;                           //  ...and posterior)
      for (j=0; j<fw->FToutLength; ++j){           // (update residuals)
        fw->AresidualAcc[j] = fw->AresidualPro[j];
        fw->EresidualAcc[j] = fw->EresidualPro[j];
      //}
      ++acceptCount;
    }
    		if(rate){
    		if (( i > 0) && ((i%100)==0)){
		acceptRate = (double)acceptCount/(double)i; // rate!
		if(acceptRate < 0.23) adj /= 1.5; else adj=adj;	
		if(acceptRate > 0.40) adj *= 1.5; else adj=adj; 
		} }
    // The Gibbs step:
    // draw a new conditional spectrum
    // (changes value of `logpost' as well!).
    drawspectrum(fw, tempvec[OwnTempIndex]);
    // update prior & likelihood:
    lprior = logprior(state, fw);	  
    loglh = likelihood(fw, fw->AresidualAcc, fw->EresidualAcc);
    /* ==== END OF `REGULAR' METROPOLIS-HASTINGS ==== */
    // --->  now attempt a swap...
    // Gather all chains' likelihood values in process #0:
    MPI_Gather(&loglh, 1, MPI_DOUBLE,
               LLHVec, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    if (MPIrank == 0){
      /* pick two chains so that  temp(swap1) < temp(swap2);  */
      /* ( swap1 = cool  /  swap2 = hot )                     */
      j = ignuin(0,MPIsize-2); // temperature index: j & j+1 are tried to swap.
      // now determine the process indices corresponding to temperatures j and j+1:
      swap1 = 0;
      while (TempIndVec[swap1] != j) ++swap1;
      swap2 = 0;
      while (TempIndVec[swap2] != j+1) ++swap2;
      ++Nproposed[j];
      // `swap1' and `swap2' are process ranks!
      // determine acceptance probability for swap:
      logalpha = LLHVec[swap1] - LLHVec[swap2];
      logalpha *= (1.0/tempvec[TempIndVec[swap2]] - 1.0/tempvec[TempIndVec[swap1]]);
      
//	if(i>dataend){
		if(!accept){
			if ((logalpha >= 0.0) || (log(genunf(0.0, 1.0)) < logalpha)){ // SWAP!
				++Naccepted[j];
				j = TempIndVec[swap1];
				TempIndVec[swap1] = TempIndVec[swap2];
				TempIndVec[swap2] = j;
			}
		} 
//	}
      // ...update logfile:
      if (((i+1) % 100) == 0){
        time(&now);
        sprintf(completefilename, "%s/%s-log.txt", outputpath, outputfile);
        output = fopen(completefilename,"w");
        fprintf(output, "%d %.0f", i+1, difftime(now, starttime));
        for (j=0; j<(MPIsize-1); ++j) fprintf(output, " %d", Nproposed[j]);
        for (j=0; j<(MPIsize-1); ++j) fprintf(output, " %d", Naccepted[j]);
        fprintf(output, "\n");
        fclose(output);
      }

    }
    // every process remember previous temperature:
    prevTemp = OwnTempIndex;
    // distribute the new temperature assignments:
    MPI_Scatter(TempIndVec, 1, MPI_INT,
                &OwnTempIndex, 1, MPI_INT,
                0, MPI_COMM_WORLD);   
    // now processes involved in swap update their (conditional) spectrum:
    if (OwnTempIndex != prevTemp){
      drawspectrum(fw, tempvec[OwnTempIndex]);
      lprior = logprior(state, fw);
      loglh = likelihood(fw, fw->AresidualAcc, fw->EresidualAcc);
    }
	/*SNR*/
	  if(i%10==0)
	{
  	snr = signaltonoiseratio(fw, state);
	}
    /* ==== END OF SWAPPING STEP ==== */
    // --->  now log to file(s)...
    if ((i % thinout) == 0){
      sprintf(completefilename, "%s/%s-C%d.txt", outputpath, outputfile, OwnTempIndex+1);
      output = fopen(completefilename,"a");
  	fprintf(output, "%d %.12f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.4f %.4f %d %.4f\n",
          i, state[lognuz], state[logmu], state[logM], state[ez], state[gamz], state[phiz], state[colati], 
          state[longi], state[Lambda], state[alpz], state[SMBHSpin], state[thetaSpin], state[phiSpin], state[logDl], 
          lprior+loglh, loglh, acceptCount, snr);
       fclose(output);
    }
	
	/* Store data row by row as data matrix for calculating covariance matrix after 100 iterations */
	
	//datamatrix(i, state, tempvec[OwnTempIndex], datamat);
	//covmat(i, datamat, datasize, tempvec[OwnTempIndex]);
	
	// update & log the (conditional) spectrum;
    // process w/ temperature =1 sends residual vector to process #0,
    // and process #0 updates its estimate and/or logs to file:
    if ((i % SpectrumUpdateSkip)==0){
      if (OwnTempIndex==0){
        for (j=0; j<fw->FToutLength; ++j)
          residualSSQ[j] = (fw->AresidualAcc[j] + fw->EresidualAcc[j]) * DFT2AB;
        if (MPIrank != 0)
          MPI_Send(residualSSQ, fw->FToutLength, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
      }
      if (MPIrank == 0){
        whichTempOne = 0;
        while (TempIndVec[whichTempOne] != 0) ++whichTempOne;
        if (whichTempOne != 0)
          MPI_Recv(residualSSQ, fw->FToutLength, MPI_DOUBLE, whichTempOne, i, MPI_COMM_WORLD, &mpistat);
        n = i / SpectrumUpdateSkip;
        weight1 = ((double)(n-1)) / ((double)n);
        weight2 = 1.0 / ((double)n);
        weight3 = (n>1) ? ((double)(n-2))/((double)(n-1)) : 0.0;
        for (j=0; j<fw->FToutLength; ++j){
          logspectrum = ((fw->Asigma[j]));//+residualSSQ[j]) / (2.0);
          logspectrum = sigmasq2spec + log(logspectrum);
	  //if (j<=10) printf(" : %d -- %d %.2f\n", i, j, logspectrum);
          //diff = logspectrum-logSpectrumMean[j];
          logSpectrumMean[j] = weight1*logSpectrumMean[j] + weight2*logspectrum;
          //logSpectrumVar[j]  = weight3*logSpectrumVar[j]  + weight1*diff*diff;
        }
      }
    }
    if ((MPIrank == 0) && (((i % SpectrumLogSkip)==0) || (i==100))){
      sprintf(completefilename, "%s/%s-spectrum.txt", outputpath, outputfile);
      output = fopen(completefilename,"w");
      fprintf(output, "%d %d", i, i/SpectrumUpdateSkip);
      for (j=1; j<(fw->FToutLength-1); ++j)
        fprintf(output, " %.6e", logSpectrumMean[j]);
      fprintf(output, "\n");
      //fprintf(output, "%d %d", i, i/SpectrumUpdateSkip);
      //for (j=1; j<(fw->FToutLength-1); ++j)
      //  fprintf(output, " %.6e", logSpectrumVar[j]);
      //fprintf(output, "\n");
      fclose(output);
    }
    /* ==== END OF ITERATION ==== */
  }
}


void printtime()
/* prints time (& date) to screen */
{
  time_t tm;
  struct tm *ltime;
  time( &tm );
  ltime = localtime( &tm );
  ltime->tm_mon++;
  ltime->tm_year += 1900;
  printf( "%02i.%02i.%04i  %02i:%02i:%02i\n\0", ltime->tm_mday, ltime->tm_mon,
          ltime->tm_year, ltime->tm_hour, ltime->tm_min, ltime->tm_sec);
}

/*-- MAIN BODY --*/

main(int argc, char *argv[])
{
 int waqas,size, rank;
  char          completefilename[200];
  TimeSeries    *ts=NULL;
  TDIframework  tdiframe;
  int           i, j, n=5;
  time_t        starttime, endtime;
  double        seconds;
  //double        theta[7];
  double        theta[14];
  double        m1,m2;
  
srand(time(NULL));// new line by me
  long random1 = rand();// editted by me
  long random2 = rand();// editted by me
 
 double zeromean[14];
  double dummycov[196];

MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // set random seed:
  setall(random1+rank*10000, random2+(size+rank)*100000);//
  for (i=0; i<10*(rank+2); ++i) seconds = genunf(0.0,1.0);
  
  if (rank == 0){
    printf(" +---One Week Data----\n", LENGTH_OF_CHOPPED);
    printf(" | This is process #%d of %d\n", rank, size);
    printf(" | initialising  :  "); printtime();
  }

  // set covariances for correlated proposals:
  /* NB These are missing and must be filled in!! */

  for (i=0; i<14; ++i){
    zeromean[i] = 0.0;
    for (j=0; j<14; ++j)
      dummycov[i*14+j] = PropCov[i][j];
  }
  setgmn(zeromean, dummycov, 14, MVNpar);


  /*-- read in data:  --*/
  sprintf(completefilename, "%s/%s", datapath, datafile);
  //printf(" | opening file: `%s'...\n", datafile);
  //fprintf(stderr," | getTDIdata(): ");
  ts = getTDIdata(completefilename);
  fixTimeSeries(ts);

  //if (ts != NULL) printf(" | ...successful!\n");
  //else printf(" | ERROR!\n");
  //chopTimeSeries(ts, 0.0, 3932160.0);
  //chopTimeSeries(ts, -HUGE_VAL, pow(2, LENGTH_OF_CHOPPED)*15);
  chopTimeSeries(ts, -HUGE_VAL, 63072000.0);       /*-- print some key figures: --*/
  printf(" | %d entries for %d variables (%s)\n", ts->Length, ts->Records, ts->Name);
  printf(" | every %g seconds; time offset: %.2e, first entry at %.1f s\n", 
         ts->Cadence, ts->TimeOffset, ts->Data[0]->data[0]);
  seconds = ts->Cadence * ts->Length;
  printf(" | (%.1f days = %.3f years of data)\n", seconds/86400.0, seconds/31557600.0);
  
//* 1B.3.2 Train */

  theta[lognuz] =log(genunf(.0002, .0005)); //freq
  theta[logmu] = log(genunf(.5,100));  //mu
  theta[logM] = log(2.5*1000000.0000);  //M
  theta[ez] = genunf(.15,.25);  //ez
  theta[gamz] = genunf(0,twopi); //gamz
  theta[phiz] = genunf(0,twopi);  //phiz
  theta[colati] = 90 -40.86638889; //colati
  theta[longi] = 10.67541667; //longi
  theta[Lambda] = genunf(0,twopi);  //Lambda
  theta[alpz] =  genunf(0,twopi); //alpz
  theta[SMBHSpin] =  genunf(0,1); //Spin
  theta[thetaSpin] =  genunf(0,twopi);   //thetaSpin
  theta[phiSpin] = genunf(0,twopi);  //phiSpin
  theta[logDl] = log(.77*(1+0.0001779795468057455)); // D= Dl(1+z)


  /*-- derive A/E/T: --*/
  //printf(" | deriving TDI combinations A/E/T...\n");
  TDIcombi(ts, 1, 1, 1);

  /*-- (pre-) initialise: --*/
  //printf(" | initialising `TDI framework' (might take several minutes)...\n");
  //time(&starttime);
  TDIinit(&tdiframe, ts, 1);
  //time(&endtime);
  //printf(" | ...finished after %.0f seconds.\n", difftime(endtime, starttime));
/*
 if (1 != 1){
    //-- inject signal: --
    printf("-+--> INJECTING SIGNAL...\n");
    double *Xresponse, *Yresponse, *Zresponse;
    double *plusInj, *crossInj;
    long k;
    double deltat=15;
    plusInj  = (double*) malloc((tdiframe.dataN +100) * sizeof(double));
    crossInj = (double*) malloc((tdiframe.dataN +100) * sizeof(double));
    GenBCWaveStart(&tdiframe, theta, plusInj, crossInj); 

    for (k=0; k<ts->Length; ++k){
      //  Inject signal into X/Y/Z channels:
      ts->Data[4]->data[k] = plusInj[k];
      ts->Data[5]->data[k] = crossInj[k];
    }	

    free(plusInj);
    free(crossInj);

    //-- clean up: --
    freeTDIframework(&tdiframe);

    //-- (re-) initialise: --
    printf(" | initialising `TDI framework' (might take several minutes)...\n");
    time(&starttime);
    TDIinit(&tdiframe, ts, 1);
    //TDIinit(&tdiframe, ts, 0);
    time(&endtime);
    printf(" | ...finished after %.0f seconds.\n", difftime(endtime, starttime));
    printf("-+--> INJECTION DONE.\n");
 }
*/
  freeTimeSeries(ts);
  //if (rank == 0) TDIsummary(&tdiframe);


  /*-- output SNR: --*/
  //printf(" |\n |   SNR: %.3f\n |\n", signaltonoiseratio(&tdiframe,theta));


  /*-- MCMC: --*/
  //printf(" | #%d: starting `metropolis()'...\n",rank); 
  metropolis(&tdiframe, theta, 1, size, rank);
  theta[SMBHSpin]=0;
  //printf(theta);
  metropolis(&tdiframe, theta, 1, size, rank); //1000000);


  /*-- clean up: --*/
  freeTDIframework(&tdiframe);
  MPI_Finalize();

  if (rank == 0){
    printf(" +------------------------------------------------------------------\n");
  }
  return 0;
}
