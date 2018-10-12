gr-tensorflow_CC
----------------

A GNU Radio example module to demonstrate the usage (inference) of trained and saved (e.g. in python) Tensorflow models in a C++ GNU Radio block.

Prerequisites
-------------

Obviously a working GNU Radio installation is required.
The more tricky part is to get the C++ part of tensorflow running to perform the inference that we want to do in the GNU Radio block. For that use [https://github.com/johschmitz/tensorflow_cc_cmake](https://github.com/johschmitz/tensorflow_cc_cmake) and follow the instructions to install tensorflow_cc.

How to build gr-tensorflow_cc
-----------------------------

The GNU Radio module can now be build.
It includes a cmake module that is compatible with the tensorflow_cc install script.

    mkdir build
    cd build
    cmake ../
    make

How to train and run the example
--------------------------------

The example models can be found in the examples folder

    cd ../examples

Training requires a working python Tensorflow environment.
I assume that Anaconda is used

    conda activate tensorflow

Then train and the example model

    ./train_autoencoder.py

Close the plots to complete the script, then save the encoder and decoder models separately for inference

    ./save_encoder_tf.py
    ./save_decoder_tf.py

Leave the Anaconda environment

    conda deactivate

Make sure your environment variables are set correctly after leaving the conda environment

    source ~/.bashrc

Or simply use a second terminal.
To avoid a full installation of the out-of-tree module you can set some additional environment variables with

    . environment_debug

you can now run the example models in GNU Radio Companion

    gnuradio-companion neural_transceiver.grc
