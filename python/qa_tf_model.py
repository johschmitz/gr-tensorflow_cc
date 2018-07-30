#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2018 Johannes Schmitz
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

from gnuradio import gr, gr_unittest
from gnuradio import blocks
import tensorflow_cc_swig as tf_cc
import numpy as np
import os

class qa_tf_model (gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown (self):
        self.tb = None

    def test_001_t (self):
        # set up fg
        num_two_bits = 1
        self.src = blocks.vector_source_f([0, 1, 1, 0]*num_two_bits,vlen=2,repeat=False)
        model_path = os.getcwd()+'/../examples/export/encoder/tf_model.meta'
        self.tf = tf_cc.tf_model(model_path,'input',4,2,'output',4,2)
        self.sink = blocks.vector_sink_f(vlen=2)
        self.tb.connect(self.src, (self.tf,0))
        self.tb.connect((self.tf,0), self.sink)
        self.tb.run()
        # check data
        result_data = self.sink.data()
        expected_result=np.linalg.norm(np.array([[1/np.sqrt(2),1/np.sqrt(2)]]*2*num_two_bits),axis=1)
        result_norm =np.linalg.norm(np.array(result_data).reshape([2*num_two_bits,2]),axis=1)
        print(expected_result)
        print(result_norm)
        self.assertFloatTuplesAlmostEqual(expected_result, result_norm, 1)

if __name__ == '__main__':
    gr_unittest.run(qa_tf_model, "qa_tf_model.xml")
