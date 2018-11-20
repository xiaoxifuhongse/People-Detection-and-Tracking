#include "libSVM.h"
#include "value.h"
#include "index.h"
#include "indices.h"
#include "coef.h"


CSVM::CSVM()
{
	model = Malloc(struct svm_model, 1);
	model->rho   = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV   = NULL;
	model->free_sv = 1;
	model->sv_indices = NULL;

 }
void CSVM::libsvmreadmodel()
{
	// 1
	model->param.svm_type = 0;
	model->param.kernel_type = 2;
	model->param.degree = 3;
	model->param.gamma = 0.1768;
	model->param.coef0 = .0;
	//2
	model->nr_class = 2;
	//3
	model->l = 6710;
	//rho
	int n = model->nr_class * (model->nr_class-1)/2;
	model->rho =(double*)malloc(n*sizeof(double));
	model->rho[0] = -0.084042275527526;

	//label
	model->label = (int*)malloc(model->nr_class*sizeof(int));
	//for(i=0;i<model->nr_class;i++)
	model->label[0] = 0;
	model->label[1] = 1;
   //sv_indices
	model->sv_indices = (int*) malloc(model->l*sizeof(int));
	for(int i =0;i<model->l; i++)
	{
	  model->sv_indices[i] = indices[i];
	}
	//probA
	model->probA = (double*) malloc(n*sizeof(double));
	model->probA[0] = -3.105162532662727;

	//probB	0.274910358363906
	model->probB = (double*) malloc(n*sizeof(double));
	model->probB[0] = 0.274910358363906;
	//nSV
	model->nSV = (int*) malloc(model->nr_class*sizeof(int));
	model->nSV[0] = 3453;
	model->nSV[1] = 3257;
	//sv_coef
	model->sv_coef = (double**) malloc((model->nr_class-1)*sizeof(double));
	for( int i=0 ; i< model->nr_class -1 ; i++ )
	{
		model->sv_coef[i] = (double*) malloc((model->l)*sizeof(double));
	}
	int cnum = 0;
	for(int i = 0; i < model->nr_class - 1; i++)
	{  
		for(int j = 0; j < model->l; j++)
		{ 
			model->sv_coef[i][j] = coef[cnum++];

		}
	}
   //SVs
	  {
		  int sr = 6710;
		  int elements = 502025+sr;
		  model->SV = (struct svm_node**)malloc(sr*sizeof(struct svm_node*));
		  struct svm_node *x_space = (struct svm_node*)malloc(elements*sizeof(struct svm_node));
		  int low = 0;
		  int cnt = 0;
		  for(int i=0; i<sr; i++)
		  {
			  
			  int high = low+ id[i];
			  
			  int x_index = 0;
			  model->SV[i] =&x_space[low + i];
			  
			  for(int j=low; j<high; j++)
			  {
				  model->SV[i][x_index].index = index[cnt]; 
				  model->SV[i][x_index].value = value[cnt];
				  cnt++;
				  x_index++;
			  }
			   model->SV[i][x_index].index = -1;
			   low = high;
		  }
	  
	  }


}

void CSVM::libpreidctfast(Mat features, Mat& predict_estimates )
{
	

	predict(features, predict_estimates);
	
}

void CSVM::predict(Mat features, Mat& predict_estimates)
{
	int feature_number, testing_instance_number;
	int instance_index;
	double *ptr_instance;
	double *ptr_prob_estimates;

	int svm_type = svm_get_svm_type(model);
	int nr_class = svm_get_nr_class(model);

	feature_number = features.rows ;		
	testing_instance_number = features.cols ;
	
	ptr_instance = (double*)features.data;
	predict_estimates = Mat::zeros(Size(testing_instance_number,nr_class), CV_64FC1);
	ptr_prob_estimates = (double*)predict_estimates.data;

	int freeInstance = 0;

	#pragma omp parallel
	{
		struct svm_node *x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
		double *prob_estimates = (double *) malloc(nr_class*sizeof(double));
		int i,base;
		double predict_label; 

		int instance_index;
		while (true) {
		#pragma omp critical 
			{
				instance_index = freeInstance;
				freeInstance++;
			}

			if (instance_index >= testing_instance_number)
				break;    

			base = feature_number*instance_index;
			for(i=0;i<feature_number;i++)
			{
				x[i].index = i+1;
				x[i].value = ptr_instance[base + i];
			}
			x[feature_number].index = -1;

			predict_label = svm_predict_probability(model, x, prob_estimates);

			for(i=0;i<nr_class;i++)
				ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
		}
		free(x);
		free(prob_estimates);
	}



}


						 
  

