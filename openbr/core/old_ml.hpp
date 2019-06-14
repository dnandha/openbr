/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OLD_ML_HPP
#define OPENCV_OLD_ML_HPP

#ifdef __cplusplus
#  include "opencv2/core.hpp"
#endif

#include "opencv2/core/core_c.h"
#include <limits.h>

#ifdef __cplusplus

#include <map>
#include <iostream>

// Apple defines a check() macro somewhere in the debug headers
// that interferes with a method definition in this header
#undef check

/****************************************************************************************\
*                               Main struct definitions                                  *
\****************************************************************************************/

/* log(2*PI) */
#define CV_LOG2PI (1.8378770664093454835606594728112)

/* columns of <trainData> matrix are training samples */
#define CV_COL_SAMPLE 0

/* rows of <trainData> matrix are training samples */
#define CV_ROW_SAMPLE 1

#define CV_IS_ROW_SAMPLE(flags) ((flags) & CV_ROW_SAMPLE)

struct CvVectors
{
    int type;
    int dims, count;
    CvVectors* next;
    union
    {
        uchar** ptr;
        float** fl;
        double** db;
    } data;
};

#if 0
/* A structure, representing the lattice range of statmodel parameters.
   It is used for optimizing statmodel parameters by cross-validation method.
   The lattice is logarithmic, so <step> must be greater than 1. */
typedef struct CvParamLattice
{
    double min_val;
    double max_val;
    double step;
}
CvParamLattice;

CV_INLINE CvParamLattice cvParamLattice( double min_val, double max_val,
                                         double log_step )
{
    CvParamLattice pl;
    pl.min_val = MIN( min_val, max_val );
    pl.max_val = MAX( min_val, max_val );
    pl.step = MAX( log_step, 1. );
    return pl;
}

CV_INLINE CvParamLattice cvDefaultParamLattice( void )
{
    CvParamLattice pl = {0,0,0};
    return pl;
}
#endif

/* Variable type */
#define CV_VAR_NUMERICAL    0
#define CV_VAR_ORDERED      0
#define CV_VAR_CATEGORICAL  1

#define CV_TYPE_NAME_ML_SVM         "opencv-ml-svm"
#define CV_TYPE_NAME_ML_KNN         "opencv-ml-knn"
#define CV_TYPE_NAME_ML_NBAYES      "opencv-ml-bayesian"
#define CV_TYPE_NAME_ML_BOOSTING    "opencv-ml-boost-tree"
#define CV_TYPE_NAME_ML_TREE        "opencv-ml-tree"
#define CV_TYPE_NAME_ML_ANN_MLP     "opencv-ml-ann-mlp"
#define CV_TYPE_NAME_ML_CNN         "opencv-ml-cnn"
#define CV_TYPE_NAME_ML_RTREES      "opencv-ml-random-trees"
#define CV_TYPE_NAME_ML_ERTREES     "opencv-ml-extremely-randomized-trees"
#define CV_TYPE_NAME_ML_GBT         "opencv-ml-gradient-boosting-trees"

#define CV_TRAIN_ERROR  0
#define CV_TEST_ERROR   1

/****************************************************************************************\
*                                   Boosted tree classifier                              *
\****************************************************************************************/


struct CvParamGrid
{
    // SVM params type
    enum { SVM_C=0, SVM_GAMMA=1, SVM_P=2, SVM_NU=3, SVM_COEF=4, SVM_DEGREE=5 };

    CvParamGrid()
    {
        min_val = max_val = step = 0;
    }

    CvParamGrid( double min_val, double max_val, double log_step );
    //CvParamGrid( int param_id );
    bool check() const;

    CV_PROP_RW double min_val;
    CV_PROP_RW double max_val;
    CV_PROP_RW double step;
};

inline CvParamGrid::CvParamGrid( double _min_val, double _max_val, double _log_step )
{
    min_val = _min_val;
    max_val = _max_val;
    step = _log_step;
}


struct CvSVMParams
{
    CvSVMParams();
    CvSVMParams( int svm_type, int kernel_type,
                 double degree, double gamma, double coef0,
                 double Cvalue, double nu, double p,
                 CvMat* class_weights, CvTermCriteria term_crit );

    CV_PROP_RW int         svm_type;
    CV_PROP_RW int         kernel_type;
    CV_PROP_RW double      degree; // for poly
    CV_PROP_RW double      gamma;  // for poly/rbf/sigmoid/chi2
    CV_PROP_RW double      coef0;  // for poly/sigmoid

    CV_PROP_RW double      C;  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    CV_PROP_RW double      nu; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    CV_PROP_RW double      p; // for CV_SVM_EPS_SVR
    CvMat*      class_weights; // for CV_SVM_C_SVC
    CV_PROP_RW CvTermCriteria term_crit; // termination criteria
};


struct CvSVMKernelRow
{
    CvSVMKernelRow* prev;
    CvSVMKernelRow* next;
    float* data;
};


struct CvSVMSolutionInfo
{
    double obj;
    double rho;
    double upper_bound_p;
    double upper_bound_n;
    double r;   // for Solver_NU
};


struct CvSVMDecisionFunc
{
    double rho;
    int sv_count;
    double* alpha;
    int* sv_index;
};


/****************************************************************************************\
*                                      Decision Tree                                     *
\****************************************************************************************/\
struct CvPair16u32s
{
    unsigned short* u;
    int* i;
};


#define CV_DTREE_CAT_DIR(idx,subset) \
    (2*((subset[(idx)>>5]&(1 << ((idx) & 31)))==0)-1)


struct CvDTreeSplit
{
    int var_idx;
    int condensed_idx;
    int inversed;
    float quality;
    CvDTreeSplit* next;
    union
    {
        int subset[2];
        struct
        {
            float c;
            int split_point;
        }
        ord;
    };
};

struct CvDTreeNode
{
    int class_idx;
    int Tn;
    double value;

    CvDTreeNode* parent;
    CvDTreeNode* left;
    CvDTreeNode* right;

    CvDTreeSplit* split;

    int sample_count;
    int depth;
    int* num_valid;
    int offset;
    int buf_idx;
    double maxlr;

    // global pruning data
    int complexity;
    double alpha;
    double node_risk, tree_risk, tree_error;

    // cross-validation pruning data
    int* cv_Tn;
    double* cv_node_risk;
    double* cv_node_error;

    int get_num_valid(int vi) { return num_valid ? num_valid[vi] : sample_count; }
    void set_num_valid(int vi, int n) { if( num_valid ) num_valid[vi] = n; }
};


struct CvDTreeParams
{
    CV_PROP_RW int   max_categories;
    CV_PROP_RW int   max_depth;
    CV_PROP_RW int   min_sample_count;
    CV_PROP_RW int   cv_folds;
    CV_PROP_RW bool  use_surrogates;
    CV_PROP_RW bool  use_1se_rule;
    CV_PROP_RW bool  truncate_pruned_tree;
    CV_PROP_RW float regression_accuracy;
    const float* priors;

    CvDTreeParams();
    CvDTreeParams( int max_depth, int min_sample_count,
                   float regression_accuracy, bool use_surrogates,
                   int max_categories, int cv_folds,
                   bool use_1se_rule, bool truncate_pruned_tree,
                   const float* priors );
};


struct CvBoostParams : public CvDTreeParams
{
    CV_PROP_RW int boost_type;
    CV_PROP_RW int weak_count;
    CV_PROP_RW int split_criteria;
    CV_PROP_RW double weight_trim_rate;

    CvBoostParams();
    CvBoostParams( int boost_type, int weak_count, double weight_trim_rate,
                   int max_depth, bool use_surrogates, const float* priors );
};


struct CvGBTreesParams : public CvDTreeParams
{
    CV_PROP_RW int weak_count;
    CV_PROP_RW int loss_function_type;
    CV_PROP_RW float subsample_portion;
    CV_PROP_RW float shrinkage;

    CvGBTreesParams();
    CvGBTreesParams( int loss_function_type, int weak_count, float shrinkage,
        float subsample_portion, int max_depth, bool use_surrogates );
};


struct CvDTreeTrainData
{
    CvDTreeTrainData();
    CvDTreeTrainData( const CvMat* trainData, int tflag,
                      const CvMat* responses, const CvMat* varIdx=0,
                      const CvMat* sampleIdx=0, const CvMat* varType=0,
                      const CvMat* missingDataMask=0,
                      const CvDTreeParams& params=CvDTreeParams(),
                      bool _shared=false, bool _add_labels=false );

    int sample_count, var_all, var_count, max_c_count;
    int ord_var_count, cat_var_count, work_var_count;
    bool have_labels, have_priors;
    bool is_classifier;
    int tflag;

    const CvMat* train_data;
    const CvMat* responses;
    CvMat* responses_copy; // used in Boosting

    int buf_count, buf_size; // buf_size is obsolete, please do not use it, use expression ((int64)buf->rows * (int64)buf->cols / buf_count) instead
    bool shared;
    int is_buf_16u;

    CvMat* cat_count;
    CvMat* cat_ofs;
    CvMat* cat_map;

    CvMat* counts;
    CvMat* buf;
    inline size_t get_length_subbuf() const
    {
        size_t res = (size_t)(work_var_count + 1) * (size_t)sample_count;
        return res;
    }

    CvMat* direction;
    CvMat* split_buf;

    CvMat* var_idx;
    CvMat* var_type; // i-th element =
                     //   k<0  - ordered
                     //   k>=0 - categorical, see k-th element of cat_* arrays
    CvMat* priors;
    CvMat* priors_mult;

    CvDTreeParams params;

    CvMemStorage* tree_storage;
    CvMemStorage* temp_storage;

    CvDTreeNode* data_root;

    CvSet* node_heap;
    CvSet* split_heap;
    CvSet* cv_heap;
    CvSet* nv_heap;

    cv::RNG* rng;
};

/****************************************************************************************\
*                                      Data                                             *
\****************************************************************************************/

#define CV_COUNT     0
#define CV_PORTION   1

struct CvTrainTestSplit
{
    CvTrainTestSplit();
    CvTrainTestSplit( int train_sample_count, bool mix = true);
    CvTrainTestSplit( float train_sample_portion, bool mix = true);

    union
    {
        int count;
        float portion;
    } train_sample_part;
    int train_sample_part_mode;

    bool mix;
};


#endif // __cplusplus
#endif // OPENCV_OLD_ML_HPP

/* End of file. */
