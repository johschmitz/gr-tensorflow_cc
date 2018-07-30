/* -*- c++ -*- */

#define TENSORFLOW_CC_API

%include "gnuradio.i" // the common stuff

//load generated python docstrings
%include "tensorflow_cc_swig_doc.i"

%{
#include "tensorflow_cc/tf_model.h"
#include "tensorflow_cc/throughput_measure.h"
%}


%include "tensorflow_cc/tf_model.h"
GR_SWIG_BLOCK_MAGIC2(tensorflow_cc, tf_model);
%include "tensorflow_cc/throughput_measure.h"
GR_SWIG_BLOCK_MAGIC2(tensorflow_cc, throughput_measure);
