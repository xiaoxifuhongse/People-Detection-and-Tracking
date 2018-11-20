#pragma once
#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
 class CSVM
 {
 public:
	 CSVM();
	 ~CSVM(){};
 public:

	 void libsvmreadmodel();
	 void libpreidctfast(Mat features, Mat &predict_prob);
 protected:
	 void predict(Mat features,Mat &predict_estimates);
	
 public:
	 struct svm_model *model;

 };