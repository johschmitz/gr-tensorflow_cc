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


#ifndef INCLUDED_TENSORFLOW_CC_THROUGHPUT_MEASURE_H
#define INCLUDED_TENSORFLOW_CC_THROUGHPUT_MEASURE_H

#include <tensorflow_cc/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
  namespace tensorflow_cc {

    /*!
     * \brief <+description of block+>
     * \ingroup tensorflow_cc
     *
     */
    class TENSORFLOW_CC_API throughput_measure : virtual public gr::sync_block
    {
     public:
      typedef boost::shared_ptr<throughput_measure> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of tensorflow_cc::throughput_measure.
       *
       * To avoid accidental use of raw pointers, tensorflow_cc::throughput_measure's
       * constructor is in a private implementation
       * class. tensorflow_cc::throughput_measure::make is the public interface for
       * creating new instances.
       */
      static sptr make(size_t itemsize, size_t vlen);
    };

  } // namespace tensorflow_cc
} // namespace gr

#endif /* INCLUDED_TENSORFLOW_CC_THROUGHPUT_MEASURE_H */

