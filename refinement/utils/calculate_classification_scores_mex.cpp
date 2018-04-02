/**
 * Copyright 2011,2013 B. Schauerte. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are 
 * met:
 * 
 *    1. Redistributions of source code must retain the above copyright 
 *       notice, this list of conditions and the following disclaimer.
 * 
 *    2. Redistributions in binary form must reproduce the above copyright 
 *       notice, this list of conditions and the following disclaimer in 
 *       the documentation and/or other materials provided with the 
 *       distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 * DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *  
 * The views and conclusions contained in the software and documentation
 * are those of the authors and should not be interpreted as representing 
 * official policies, either expressed or implied, of B. Schauerte.
 */

/**
 * If you use any of this work in scientific research or as part of a larger
 * software system, you are kindly requested to cite the use in any related 
 * publications or technical documentation. The work is based upon:
 *
 * [1] B. Schauerte, R. Stiefelhagen, "How the Distribution of Salient
 *     Objects in Images Influences Salient Object Detection". In Proceedings
 *     of the 20th International Conference on Image Processing (ICIP), 2013.
 */

#include <cmath>

#ifdef __MEX
#define __CONST__ const
#include "mex.h"
#include "matrix.h"
#endif

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

template <typename T, typename S>
void
CalculateClassficationStatistics(const T* array, const S* mask, int numel, const T threshold,
        double& precision, double& recall, double& tnr, double& accuracy, double& fscore, // precision/recall/true-negative-rate/accuracy/f-score
        unsigned int& tp, unsigned int& tn, unsigned int& fp, unsigned int& fn,           // true positives/true negatives/false positives/false negatives
        const S mask_threshold = 0, const double fscore_beta = 1,
        const double pr_div_by_zero_result = 1)
{  
  // basic
  tp = 0; // number of true positives
  tn = 0; // number of true negatives
  fp = 0; // number of false positives
  fn = 0; // number of false negatives
  
  for (int i = 0; i < numel; i++)
  {
    const T aval = array[i];
    const S mval = mask[i];
    
    if (aval >= threshold)
    {
      if (mval > mask_threshold)
        ++tp;
      else
        ++fp;
    }
    else
    {
      if (mval > mask_threshold)
        ++fn;
      else
        ++tn;
    }
  }
  
  //mexPrintf("%f\n",pr_div_by_zero_result);
  
  // calculate recall/precision/true-negative-rate/accuracy/f-score
  if (tp != 0)
  {
    precision = (double)(tp) / (double)(tp + fp);                // precision 
    recall    = (double)(tp) / (double)(tp + fn);                // recall 
  }
  else
  {
    // handle possible extreme cases (see http://stats.stackexchange.com/questions/1773/what-are-correct-values-for-precision-and-recall-in-edge-cases)
    //   Recall = 1 when FN=0, since 100% of the TP were discovered
    //   Precision = 1 when FP=0, since no there were no spurious results
    // However, software such as Weka and the evaluation implementation by
    // Cheng et al. ("Global contrast based salient region detection",
    // CVPR, 2011) do it different, i.e. they return 0 instead of 1.
//     precision = (fp > 0 ? ((double)(tp) / (double)(tp + fp)) : 1);
//     recall    = (fn > 0 ? ((double)(tp) / (double)(tp + fn)) : 1);
    precision = (fp > 0 ? ((double)(tp) / (double)(tp + fp)) : pr_div_by_zero_result);
    recall    = (fn > 0 ? ((double)(tp) / (double)(tp + fn)) : pr_div_by_zero_result);
  }
  tnr         = (double)(tn) / (double)(tn + fp);                // true negative rate 
  accuracy    = (double)(tp + tn) / (double)(tp + tn + fp + fn); // accuracy
  
  if ((precision * recall) == 0)
    fscore    = 0;
  else
    fscore    = (1 + SQR(fscore_beta)) * (precision * recall) / ((SQR(fscore_beta) * precision) + recall); // f-score
}

#ifdef __MEX
template <typename T, typename S>
void
_mexFunction(int nlhs, mxArray* plhs[],
             int nrhs, const mxArray* prhs[])
{
  __CONST__ mxArray *mindata = prhs[0]; // input array
  __CONST__ mxArray *mmask = prhs[1];   // ground-truth mask
  
  if (mxIsComplex(mindata))
    mexErrMsgTxt("only real data allowed");

  // get the number of elements
  const size_t numel = mxGetNumberOfElements(mindata);
  if (numel != mxGetNumberOfElements(mmask))
    mexErrMsgTxt("the input array and mask need to have the same number of elements");
  
  // directly get the threshold
  const T threshold = (T)(nrhs > 2 ? (T)mxGetScalar(prhs[2]) : 0); // classification threshold for the input array
  const S mask_threshold = (S)(nrhs > 3 ? (T)mxGetScalar(prhs[3]) : 0); // classification threshold for the mask
  double fscore_beta = (double)(nrhs > 4 ? (T)mxGetScalar(prhs[4]) : 1); // F-score/F-measure beta value
  double pr_div_by_zero_result = (double)(nrhs > 5 ? (T)mxGetScalar(prhs[5]) : 1); // result of precision/recall in case of (tp + fp)==0 or (tp + fn)==0

  // create the output data
  mxArray *mprecision = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *mrecall    = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *mtnr       = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *maccuracy  = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *mfscore    = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *mtp        = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *mtn        = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *mfp        = mxCreateDoubleMatrix(1,1,mxREAL);
  mxArray *mfn        = mxCreateDoubleMatrix(1,1,mxREAL);
  
  // get the real data pointers
  __CONST__ T* indata=(T*)mxGetData(mindata);
  __CONST__ S* mask=(S*)mxGetData(mmask);
  
  unsigned int tmp_tp, tmp_tn, tmp_fp, tmp_fn;
  
  // calculate the values
  CalculateClassficationStatistics(indata,mask,numel,threshold,
          mxGetPr(mprecision)[0],mxGetPr(mrecall)[0],mxGetPr(mtnr)[0],mxGetPr(maccuracy)[0],mxGetPr(mfscore)[0],
          tmp_tp, tmp_tn, tmp_fp, tmp_fn,
          mask_threshold,fscore_beta,
          pr_div_by_zero_result);
  mxGetPr(mtp)[0] = (double)tmp_tp;
  mxGetPr(mtn)[0] = (double)tmp_tn;
  mxGetPr(mfp)[0] = (double)tmp_fp;
  mxGetPr(mfn)[0] = (double)tmp_fn;
  
  // set output variables (plhs) and de-allocate unused memory
  if (nlhs < 1)
    mxDestroyArray(mprecision);
  else
    plhs[0] = mprecision;
  if (nlhs < 2)
    mxDestroyArray(mrecall);
  else
    plhs[1] = mrecall;
  if (nlhs < 3)
    mxDestroyArray(mtnr);
  else
    plhs[2] = mtnr;
  if (nlhs < 4)
    mxDestroyArray(maccuracy);
  else
    plhs[3] = maccuracy;
  if (nlhs < 5)
    mxDestroyArray(mfscore);
  else
    plhs[4] = mfscore;
  if (nlhs < 6)
    mxDestroyArray(mtp);
  else
    plhs[5] = mtp;
  if (nlhs < 7)
    mxDestroyArray(mtn);
  else
    plhs[6] = mtn;
  if (nlhs < 8)
    mxDestroyArray(mfp);
  else
    plhs[7] = mfp;
  if (nlhs < 9)
    mxDestroyArray(mfn);
  else
    plhs[8] = mfn;
}

void
mexFunction(int nlhs, mxArray* plhs[],
            int nrhs, const mxArray* prhs[])
{
  // check number of input parameters
  if (nrhs < 3 || nrhs > 6)
    mexErrMsgTxt("input arguments: array binary_mask threshold [mask_threshold fscore_beta pr_div_by_zero_result]");
  
  // output order is: precision recall tnr accuracy fscore tp tn fp fn

  // only float and double are currently supported
  if (!mxIsDouble(prhs[0]) && !mxIsSingle(prhs[0])) 
  	mexErrMsgTxt("Only float and double are supported.");
  
  // for code simplicity: the input array and mask need to be of the same type
  if (mxGetClassID(prhs[0]) != mxGetClassID(prhs[1]))
    mexErrMsgTxt("The array and the mask need to have the same data type.");
  
  switch (mxGetClassID(prhs[0]))
  {
    case mxSINGLE_CLASS:
      _mexFunction<float,float>(nlhs,plhs,nrhs,prhs);
      break;
    case mxDOUBLE_CLASS:
      _mexFunction<double,double>(nlhs,plhs,nrhs,prhs);
      break;
    default:
      // this should never happen
      break;
  }
}
#endif