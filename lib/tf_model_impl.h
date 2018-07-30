/* -*- c++ -*- */
/* 
 * Copyright 2018 Johannes Schmitz.
 * 
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 * 
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_TENSORFLOW_CC_TF_MODEL_IMPL_H
#define INCLUDED_TENSORFLOW_CC_TF_MODEL_IMPL_H

#include <tensorflow_cc/tf_model.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>


namespace gr {
    namespace tensorflow_cc {

        class tf_model_impl : public tf_model
        {
            private:
                std::string d_layer_in;
                size_t d_itemsize_in;
                size_t d_vlen_in;
                std::string d_layer_out;
                size_t d_itemsize_out;
                size_t d_vlen_out;
                bool d_use_gpu;
                tensorflow::DataType d_dtype_in;
                tensorflow::DataType d_dtype_out;
                tensorflow::Session *d_session;

            public:
                tf_model_impl(std::string model_meta_path,
                              std::string d_layer_in, size_t itemsize_in, size_t vlen_in,
                              std::string d_layer_out, size_t itemsize_out, size_t vlen_out,
                              bool use_gpu);
                ~tf_model_impl();

                tensorflow::Status load_tf_model(tensorflow::Session *sess,
                                                 std::string graph_fn,
                                                 std::string checkpoint_fn);
                int work(int noutput_items,
                        gr_vector_const_void_star &input_items,
                        gr_vector_void_star &output_items);
        };

    } // namespace tensorflow_cc
} // namespace gr

#endif /* INCLUDED_TENSORFLOW_CC_TF_MODEL_IMPL_H */

