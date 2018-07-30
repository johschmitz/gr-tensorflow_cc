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

#ifndef INCLUDED_TENSORFLOW_CC_THROUGHPUT_MEASURE_IMPL_H
#define INCLUDED_TENSORFLOW_CC_THROUGHPUT_MEASURE_IMPL_H

#include <tensorflow_cc/throughput_measure.h>

namespace gr {
  namespace tensorflow_cc {

    class throughput_measure_impl : public throughput_measure
      {
          private:
              size_t d_itemsize;
              size_t d_vlen;
              bool   d_init;
              struct timeval d_start;
              double d_total_samples;
              double d_min;
              double d_max;
              double d_avg;

          public:
              throughput_measure_impl(size_t itemsize, size_t vlen);
              ~throughput_measure_impl();

              double get_max() const  { return d_max; }
              double get_min() const  { return d_min; }
              double get_avg() const  { return d_avg; }

              int work(int noutput_items,
                      gr_vector_const_void_star &input_items,
                      gr_vector_void_star &output_items);
      };

  } // namespace tensorflow_cc
} // namespace gr

#endif /* INCLUDED_TENSORFLOW_CC_THROUGHPUT_MEASURE_IMPL_H */

