# Copyright 2019 TerraPower, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
unittests for iterables.py
"""
import time
import unittest
import numpy as np

from armi.utils import iterables

# CONSTANTS
_TEST_DATA = {"turtle": [float(vv) for vv in range(-2000, 2000)]}


class TestIterables(unittest.TestCase):
    """Testing our custom Iterables"""

    def test_flatten(self):
        self.assertEqual(
            iterables.flatten([[1, 2, 3], [4, 5, 6], [7, 8], [9, 10]]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        self.assertEqual(
            iterables.flatten([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

    def test_chunk(self):
        self.assertEqual(
            list(iterables.chunk([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4)),
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]],
        )

    def test_split(self):
        data = list(range(50))
        chu = iterables.split(data, 10)
        self.assertEqual(len(chu), 10)
        unchu = iterables.flatten(chu)
        self.assertEqual(data, unchu)

        chu = iterables.split(data, 1)
        self.assertEqual(len(chu), 1)
        unchu = iterables.flatten(chu)
        self.assertEqual(data, unchu)

        chu = iterables.split(data, 60, padWith=[None])
        self.assertEqual(len(chu), 60)
        unchu = iterables.flatten(chu)
        self.assertEqual(len(unchu), 60)

        chu = iterables.split(data, 60, padWith=[None])
        self.assertEqual(len(chu), 60)

        data = [0]
        chu = iterables.split(data, 1)
        unchu = iterables.flatten(chu)
        self.assertEqual(unchu, data)

    def test_packingAndUnpackingBinaryStrings(self):
        start = time.perf_counter()
        packed = iterables.packBinaryStrings(_TEST_DATA)
        unpacked = iterables.unpackBinaryStrings(packed["turtle"][0])
        timeDelta = time.perf_counter() - start
        self.assertEqual(_TEST_DATA["turtle"], unpacked)
        return timeDelta

    def test_packingAndUnpackingHexStrings(self):
        start = time.perf_counter()
        packed = iterables.packHexStrings(_TEST_DATA)
        unpacked = iterables.unpackHexStrings(packed["turtle"][0])
        timeDelta = time.perf_counter() - start
        self.assertEqual(_TEST_DATA["turtle"], unpacked)
        return timeDelta

    def test_isJagged(self):
        # simple list-of-lists
        rawListGood = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        self.assertFalse(iterables.isJagged(rawListGood))
        rawListBad = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 4]]
        self.assertTrue(iterables.isJagged(rawListBad))

        # simple, but one list deeper
        raw3dGood = [[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3]]]
        self.assertFalse(iterables.isJagged(raw3dGood))
        raw3dBad = [[[1], [2], [3]], [[1], [2], [3]], [[1], [2], [3, 4]]]
        self.assertTrue(iterables.isJagged(raw3dBad))

        # simple lists of 1D numpy arrays
        arrayListGood = [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])]
        self.assertFalse(iterables.isJagged(arrayListGood))
        arrayListBad = [
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
        ]
        self.assertTrue(iterables.isJagged(arrayListBad))

        # simple lists of 2D numpy arrays
        array2dListGood = [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))]
        self.assertFalse(iterables.isJagged(array2dListGood))
        array2dListBad = [np.zeros((3, 5)), np.ones((3, 6)), np.zeros((3, 5))]
        self.assertTrue(iterables.isJagged(array2dListBad))

        # lists of lists of 2D numpy arrays
        array3dListGood = [
            [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
            [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
        ]
        self.assertFalse(iterables.isJagged(array3dListGood))
        array3dListBad = [
            [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
            [np.zeros((3, 5)), np.ones((3, 6)), np.zeros((3, 5))],
        ]
        self.assertTrue(iterables.isJagged(array3dListBad))

        # lists of lists of lists of 2D numpy arrays
        array4dListGood = [
            [
                [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
                [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
            ],
            [
                [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
                [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
            ],
        ]
        self.assertFalse(iterables.isJagged(array4dListGood))
        array4dListBad = [
            [
                [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
                [np.zeros((3, 5)), np.ones((3, 6)), np.zeros((3, 5))],
            ],
            [
                [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
                [np.zeros((3, 5)), np.ones((3, 5)), np.zeros((3, 5))],
            ],
        ]
        self.assertTrue(iterables.isJagged(array4dListBad))

        # a persnickety error that can be hard to miss, if your recursion is wonky
        rawSubtleBad = [[[1], [2], [3]], [[1], [2], [3], [4]], [[1], [2], [3]]]
        self.assertTrue(iterables.isJagged(rawSubtleBad))

        # prove that we handle strings as data and not collections
        stringsGood = [["1", "2"], ["1", "2"], ["1", "2"]]
        self.assertFalse(iterables.isJagged(stringsGood))
        stringsBad = [["1", "2"], ["1", "2"], ["1", "2", "3"]]
        self.assertTrue(iterables.isJagged(stringsBad))

        # make sure we handle tuples correctly
        tuplesGoodList = [(1, 2), (3, 4), (8, 9)]
        self.assertFalse(iterables.isJagged(tuplesGoodList))
        tuplesBadList = [(1, 2), (3, 4, 5), (8, 9)]
        self.assertTrue(iterables.isJagged(tuplesBadList))

        # make sure we handle Nones correctly
        nonesList = [None, None, None, (1, 2), (4, 5, 6), (1, 2), None]
        self.assertTrue(iterables.isJagged(nonesList))

        # Can we handle mixed collection types?
        mixedTypesGood = [[1, 2, 3], (4, 5, 6)]
        self.assertFalse(iterables.isJagged(mixedTypesGood))
        mixedTypesBad = [[1, 2, 3], (4, 5, 6, 7)]
        self.assertTrue(iterables.isJagged(mixedTypesBad))
        veryMixedTypes = [[1, 2, 3], 7, (4, 5, 6)]
        self.assertTrue(iterables.isJagged(veryMixedTypes))
        veryMixedTypes2 = [1, [2, 3, 4], (7, 8, 9)]
        self.assertTrue(iterables.isJagged(veryMixedTypes2))

    def test_sequence(self):
        # sequentially using methods in the usual way
        s = iterables.Sequence(range(1000000))
        s.drop(lambda i: i % 2 == 0)
        s.select(lambda i: i < 20)
        s.transform(lambda i: i * 10)
        result = tuple(s)
        self.assertEqual(result, (10, 30, 50, 70, 90, 110, 130, 150, 170, 190))

        # stringing together the methods in a more modern Python way
        s = iterables.Sequence(range(1000000))
        result = tuple(
            s.drop(lambda i: i % 2 == 0)
            .select(lambda i: i < 20)
            .transform(lambda i: i * 10)
        )
        self.assertEqual(result, (10, 30, 50, 70, 90, 110, 130, 150, 170, 190))

        # call tuple() after a couple methods
        s = iterables.Sequence(range(1000000))
        s.drop(lambda i: i % 2 == 0)
        s.select(lambda i: i < 20)
        result = tuple(s)
        self.assertEqual(result, (1, 3, 5, 7, 9, 11, 13, 15, 17, 19))

        # you can't just call tuple() a second time, there is no data left
        s.transform(lambda i: i * 10)
        result = tuple(s)
        self.assertEqual(result, ())


if __name__ == "__main__":
    unittest.main()
