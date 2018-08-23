/* -*- c++ -*- */
/* 
 * Copyright 2018 Johannes Schmitz
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "tf_model_impl.h"
#include <assert.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>


namespace gr {
    namespace tensorflow_cc {

        tf_model::sptr
        tf_model::make(std::string model_meta_path,
                       std::string layer_in, size_t itemsize_in, size_t vlen_in,
                       std::string layer_out, size_t itemsize_out, size_t vlen_out,
                       bool use_gpu)
        {
            return gnuradio::get_initial_sptr
                (new tf_model_impl(model_meta_path,
                                   layer_in, itemsize_in, vlen_in,
                                   layer_out, itemsize_out, vlen_out,
                                   use_gpu));
        }

        /*
         * The private constructor
         */
        tf_model_impl::tf_model_impl(std::string model_meta_path,
                                     std::string layer_in, size_t itemsize_in, size_t vlen_in,
                                     std::string layer_out, size_t itemsize_out, size_t vlen_out,
                                     bool use_gpu = true)
            : gr::sync_block("tf_model",
                             gr::io_signature::make(1, 1, itemsize_in*vlen_in),
                             gr::io_signature::make(1, 1, itemsize_out*vlen_out))
            , d_layer_in(layer_in)
            , d_itemsize_in(itemsize_in)
            , d_vlen_in(vlen_in)
            , d_layer_out(layer_out)
            , d_itemsize_out(itemsize_out)
            , d_vlen_out(vlen_out)
            , d_use_gpu(use_gpu)
            , d_dtype_in(tensorflow::DT_INVALID)
            , d_dtype_out(tensorflow::DT_INVALID)
        {
            // model path, input and output layers need to be specified
            assert(!model_meta_path.empty());
            assert(!layer_in.empty());
            assert(!layer_out.empty());
            // Set the path to the model .meta file
            const std::string graph_fn = model_meta_path;
            // for checkpoint data remove ".meta"
            const std::string checkpoint_fn = model_meta_path.substr(0, model_meta_path.size()-5);

            GR_LOG_INFO(d_logger, "Loading Tensorflow model from: ");

            // session configuration
            tensorflow::SessionOptions session_options;
            if(d_use_gpu == false)
            {
                tensorflow::ConfigProto* config = &session_options.config;
                // disabled GPU entirely
                (*config->mutable_device_count())["GPU"] = 0;
                // place nodes somewhere
                config->set_allow_soft_placement(true);
            }

            // prepare session
            TF_CHECK_OK(tensorflow::NewSession(session_options, &d_session));
            TF_CHECK_OK(tf_model_impl::load_tf_model(d_session, graph_fn, checkpoint_fn));
        }

        /*
         * Our virtual destructor.
         */
        tf_model_impl::~tf_model_impl()
        {
            d_session->Close();
            delete d_session;
        }

        tensorflow::Status
        tf_model_impl::load_tf_model(tensorflow::Session *sess,
                                     std::string graph_fn,
                                     std::string checkpoint_fn = "")
        {
            tensorflow::Status status;

            // Read in the protobuf graph we exported
            tensorflow::MetaGraphDef meta_graph_def;
            status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &meta_graph_def);
            if (status != tensorflow::Status::OK())
                return status;

            // create the graph in the current session
            status = sess->Create(meta_graph_def.graph_def());
            if (status != tensorflow::Status::OK())
                return status;

            // restore model from checkpoint
            const std::string restore_op_name = meta_graph_def.saver_def().restore_op_name();
            const std::string filename_tensor_name = meta_graph_def.saver_def().filename_tensor_name();

            tensorflow::Tensor filename_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
            filename_tensor.scalar<std::string>()() = checkpoint_fn;

            std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
                {filename_tensor_name, filename_tensor}
            };
            status = sess->Run(feed_dict,
                    {},
                    {restore_op_name},
                    nullptr);
            if (status != tensorflow::Status::OK())
                return status;

            // determine data type of input and output layers from graph definition
            // FIXME: check if nodes exist
            for (int i = 0; i < meta_graph_def.graph_def().node_size(); i++) {
                if (meta_graph_def.graph_def().node(i).name() == d_layer_in) {
                    auto attr = meta_graph_def.graph_def().node(i).attr();
                    d_dtype_in = attr["dtype"].type();
                }
                if (meta_graph_def.graph_def().node(i).name() == d_layer_out) {
                    auto attr = meta_graph_def.graph_def().node(i).attr();
                    d_dtype_out = attr["T"].type();
                }
            }
        }

        int
        tf_model_impl::work(int noutput_items,
                gr_vector_const_void_star &input_items,
                gr_vector_void_star &output_items)
        {
            const char *in = (const char *) input_items[0];
            char *out = (char *) output_items[0];

            // prepare tensorflow inputs
            // tensor dimension 0 is the batch dimension, i.e., time dimension
            tensorflow::Tensor in_tensor(d_dtype_in, {noutput_items, (int)d_vlen_in});
            switch(d_dtype_in) {
                // implement the most relevant data types,
                // see tensorflow's types.proto for more
                case tensorflow::DT_FLOAT: {
                    auto in_tensor_data = in_tensor.flat<float>().data();
                    memcpy(in_tensor_data, in, noutput_items*d_itemsize_in*d_vlen_in);
                    break;
                }
                case tensorflow::DT_INT32: {
                    auto in_tensor_data = in_tensor.flat<int>().data();
                    memcpy(in_tensor_data, in, noutput_items*d_itemsize_in*d_vlen_in);
                    break;
                }
                case tensorflow::DT_UINT8: {
                    auto in_tensor_data = in_tensor.flat<tensorflow::uint8>().data();
                    memcpy(in_tensor_data, in, noutput_items*d_itemsize_in*d_vlen_in);
                    break;
                }
                case tensorflow::DT_INT16: {
                    auto in_tensor_data = in_tensor.flat<tensorflow::int16>().data();
                    memcpy(in_tensor_data, in, noutput_items*d_itemsize_in*d_vlen_in);
                    break;
                }
                case tensorflow::DT_INT8: {
                    auto in_tensor_data = in_tensor.flat<tensorflow::int8>().data();
                    memcpy(in_tensor_data, in, noutput_items*d_itemsize_in*d_vlen_in);
                    break;
                }
                case tensorflow::DT_COMPLEX64: {
                    auto in_tensor_data = in_tensor.flat<tensorflow::complex64>().data();
                    memcpy(in_tensor_data, in, noutput_items*d_itemsize_in*d_vlen_in);
                    break;
                }
                default: {
                    return 0;
                    break;
                }
            }

            std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
                { d_layer_in, in_tensor },
            };

            // prepare tensorflow outputs
            std::vector<tensorflow::Tensor> out_tensors;

            TF_CHECK_OK(d_session->Run(feed_dict, {d_layer_out}, {}, &out_tensors));

            // copy tensorflow output to block output
            switch(d_dtype_out) {
                // implement the most relevant data types,
                // see tensorflow's types.proto for more
                case tensorflow::DT_FLOAT: {
                    auto out_tensor_data = out_tensors[0].flat<float>().data();
                    memcpy(out, out_tensor_data, noutput_items*d_itemsize_out*d_vlen_out);
                    break;
                }
                case tensorflow::DT_INT32: {
                    auto out_tensor_data = out_tensors[0].flat<int>().data();
                    memcpy(out, out_tensor_data, noutput_items*d_itemsize_out*d_vlen_out);
                    break;
                }
                case tensorflow::DT_UINT8: {
                    auto out_tensor_data = out_tensors[0].flat<tensorflow::uint8>().data();
                    memcpy(out, out_tensor_data, noutput_items*d_itemsize_out*d_vlen_out);
                    break;
                }
                case tensorflow::DT_INT16: {
                    auto out_tensor_data = out_tensors[0].flat<tensorflow::int16>().data();
                    memcpy(out, out_tensor_data, noutput_items*d_itemsize_out*d_vlen_out);
                    break;
                }
                case tensorflow::DT_INT8: {
                    auto out_tensor_data = out_tensors[0].flat<tensorflow::int8>().data();
                    memcpy(out, out_tensor_data, noutput_items*d_itemsize_out*d_vlen_out);
                    break;
                }
                case tensorflow::DT_COMPLEX64: {
                    auto out_tensor_data = out_tensors[0].flat<tensorflow::complex64>().data();
                    memcpy(out, out_tensor_data, noutput_items*d_itemsize_out*d_vlen_out);
                    break;
                }
                default: {
                    return 0;
                    break;
                }
            }

            // Tell runtime system how many output items we produced.
            return noutput_items;
        }

    } /* namespace tensorflow_cc */
} /* namespace gr */

