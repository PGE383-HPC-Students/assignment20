#!/usr/bin/env python

# Copyright 2020-2021 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np

from PyTrilinos import Epetra
from assignment20 import OneDimNonliear


class TestOneDimNonlinear(unittest.TestCase):

    def setUp(self):

        self.comm = Epetra.PyComm()
        self.rank = self.comm.MyPID()

    def test_nonlinear(self):

        solver = OneDimNonliear(self.comm, nx=10, k=10)
        solver.solve()
        sol = solver.get_final_solution()
        if self.rank == 0:
            np.testing.assert_allclose(sol, np.array([0., 0.06673361,
                                       0.13401702, 0.20351778, 0.27813205,
                                       0.36229663, 0.462666, 0.58946252,
                                       0.75915608, 1.]), atol=0.01)


if __name__ == '__main__':
    unittest.main()
