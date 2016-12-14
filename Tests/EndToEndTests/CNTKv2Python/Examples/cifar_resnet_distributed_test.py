# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
import pytest
import subprocess
from cifar_convnet_distributed_test import mpiexec_test

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python"))
<<<<<<< HEAD
from TrainResNet_CIFAR10_Distributed import resnet_cifar10
=======
script_under_test = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ResNet", "Python", "TrainResNet_CIFAR10_Distributed.py")
>>>>>>> Adding more distributed tests for python

def test_cifar_convnet_distributed_mpiexec(device_id):
    params = [ "-e", "2"] # run only 2 epochs
    mpiexec_test(device_id, train_and_test_script, params, 0.5946, False, True)

def test_cifar_convnet_distributed_1bitsgd_mpiexec(device_id):
    params = ["-q", "1", "-e", "2"] # 2 epochs with 1BitSGD
    mpiexec_test(device_id, train_and_test_script, params, 0.5946, False, True)

<<<<<<< HEAD
    if not is_1bit_sgd:
        pytest.skip('test only runs in 1-bit SGD')

    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
    except KeyError:
        base_path = os.path.join(
            *"../../../../Examples/Image/DataSets/CIFAR-10".split("/"))

    base_path = os.path.normpath(base_path)
    os.chdir(os.path.join(base_path, '..'))

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1)
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    #force_deterministic_algorithms()
    # TODO: do the above; they lead to slightly different results, so not doing it for now

    train_data=os.path.join(base_path, 'train_map.txt')
    test_data=os.path.join(base_path, 'test_map.txt')
    mean_data=os.path.join(base_path, 'CIFAR-10_mean.xml')

    test_error = resnet_cifar10(train_data, test_data, mean_data, 'resnet20')

    expected_test_error = 0.282

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)
    distributed.Communicator.finalize()
=======
def test_cifar_convnet_distributed_blockmomentum_mpiexec(device_id):
    params = ["-b", "32000", "-e", "2"] # 2 epochs with BlockMomentum SGD using blocksize 32000
    mpiexec_test(device_id, train_and_test_script, params, 0.55, True, False)
>>>>>>> Adding more distributed tests for python
