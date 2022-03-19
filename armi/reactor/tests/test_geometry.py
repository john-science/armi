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

"""Tests the geometry (loading input) file"""
# pylint: disable=missing-function-docstring,missing-class-docstring,abstract-method,protected-access
import io
import os
import unittest

from armi.reactor import geometry
from armi.reactor.tests import test_reactors
from armi.tests import TEST_ROOT


class TestGeomType(unittest.TestCase):
    def testFromStr(self):
        # note the bonkers case and extra whitespace to exercise the canonicalization
        self.assertEqual(geometry.GeomType.fromStr("HeX"), geometry.GeomType.HEX)
        self.assertEqual(
            geometry.GeomType.fromStr("cARTESIAN"), geometry.GeomType.CARTESIAN
        )
        self.assertEqual(geometry.GeomType.fromStr(" thetaRZ"), geometry.GeomType.RZT)
        self.assertEqual(geometry.GeomType.fromStr("rZ  "), geometry.GeomType.RZ)

        with self.assertRaises(ValueError):
            geometry.GeomType.fromStr("what even is this?")

    def testLabel(self):
        gt = geometry.GeomType.fromStr("hex")
        self.assertEqual(gt.label, "Hexagonal")
        gt = geometry.GeomType.fromStr("cartesian")
        self.assertEqual(gt.label, "Cartesian")
        gt = geometry.GeomType.fromStr("rz")
        self.assertEqual(gt.label, "R-Z")
        gt = geometry.GeomType.fromStr("thetarz")
        self.assertEqual(gt.label, "R-Z-Theta")

    def testStr(self):
        for geom in {geometry.HEX, geometry.CARTESIAN, geometry.RZ, geometry.RZT}:
            self.assertEqual(str(geometry.GeomType.fromStr(geom)), geom)


class TestSymmetryType(unittest.TestCase):
    def testFromStr(self):
        # note the bonkers case and extra whitespace to exercise the canonicalization
        self.assertEqual(
            geometry.SymmetryType.fromStr("thiRd periodic ").domain,
            geometry.DomainType.THIRD_CORE,
        )
        st = geometry.SymmetryType.fromStr("sixteenth reflective")
        self.assertEqual(st.boundary, geometry.BoundaryType.REFLECTIVE)
        self.assertEqual(str(st), "sixteenth reflective")

        with self.assertRaises(ValueError):
            geometry.SymmetryType.fromStr("what even is this?")

    def testFromAny(self):
        st = geometry.SymmetryType.fromAny("eighth reflective through center assembly")
        self.assertTrue(st.isThroughCenterAssembly)
        self.assertEqual(st.domain, geometry.DomainType.EIGHTH_CORE)
        self.assertEqual(st.boundary, geometry.BoundaryType.REFLECTIVE)

        st = geometry.SymmetryType(
            geometry.DomainType.EIGHTH_CORE, geometry.BoundaryType.REFLECTIVE, True
        )
        self.assertTrue(st.isThroughCenterAssembly)
        self.assertEqual(st.domain, geometry.DomainType.EIGHTH_CORE)
        self.assertEqual(st.boundary, geometry.BoundaryType.REFLECTIVE)

        newST = geometry.SymmetryType.fromAny(st)
        self.assertTrue(newST.isThroughCenterAssembly)
        self.assertEqual(newST.domain, geometry.DomainType.EIGHTH_CORE)
        self.assertEqual(newST.boundary, geometry.BoundaryType.REFLECTIVE)

    def testBaseConstructor(self):
        self.assertEqual(
            geometry.SymmetryType(
                geometry.DomainType.SIXTEENTH_CORE, geometry.BoundaryType.REFLECTIVE
            ).domain,
            geometry.DomainType.SIXTEENTH_CORE,
        )
        self.assertEqual(
            str(
                geometry.SymmetryType(
                    geometry.DomainType.FULL_CORE, geometry.BoundaryType.NO_SYMMETRY
                ).boundary
            ),
            "",
        )

    def testLabel(self):
        st = geometry.SymmetryType(
            geometry.DomainType.FULL_CORE, geometry.BoundaryType.NO_SYMMETRY
        )
        self.assertEqual(st.domain.label, "Full")
        self.assertEqual(st.boundary.label, "No Symmetry")
        st = geometry.SymmetryType(
            geometry.DomainType.THIRD_CORE, geometry.BoundaryType.PERIODIC
        )
        self.assertEqual(st.domain.label, "Third")
        self.assertEqual(st.boundary.label, "Periodic")
        st = geometry.SymmetryType(
            geometry.DomainType.QUARTER_CORE, geometry.BoundaryType.REFLECTIVE
        )
        self.assertEqual(st.domain.label, "Quarter")
        self.assertEqual(st.boundary.label, "Reflective")
        st = geometry.SymmetryType(
            geometry.DomainType.EIGHTH_CORE, geometry.BoundaryType.REFLECTIVE
        )
        self.assertEqual(st.domain.label, "Eighth")
        st = geometry.SymmetryType(
            geometry.DomainType.SIXTEENTH_CORE, geometry.BoundaryType.REFLECTIVE
        )
        self.assertEqual(st.domain.label, "Sixteenth")

    def testSymmetryFactor(self):
        st = geometry.SymmetryType(
            geometry.DomainType.FULL_CORE, geometry.BoundaryType.NO_SYMMETRY
        )
        self.assertEqual(st.symmetryFactor(), 1.0)
        st = geometry.SymmetryType(
            geometry.DomainType.THIRD_CORE, geometry.BoundaryType.PERIODIC
        )
        self.assertEqual(st.symmetryFactor(), 3.0)
        st = geometry.SymmetryType(
            geometry.DomainType.QUARTER_CORE, geometry.BoundaryType.REFLECTIVE
        )
        self.assertEqual(st.symmetryFactor(), 4.0)
        st = geometry.SymmetryType(
            geometry.DomainType.EIGHTH_CORE, geometry.BoundaryType.REFLECTIVE
        )
        self.assertEqual(st.symmetryFactor(), 8.0)
        st = geometry.SymmetryType(
            geometry.DomainType.SIXTEENTH_CORE, geometry.BoundaryType.REFLECTIVE
        )
        self.assertEqual(st.symmetryFactor(), 16.0)

    def test_checkValidGeomSymmetryCombo(self):
        geomHex = geometry.GeomType.HEX
        geomCart = geometry.GeomType.CARTESIAN
        geomRZT = geometry.GeomType.RZT
        geomRZ = geometry.GeomType.RZ
        fullCore = geometry.SymmetryType(
            geometry.DomainType.FULL_CORE, geometry.BoundaryType.NO_SYMMETRY
        )
        thirdPeriodic = geometry.SymmetryType(
            geometry.DomainType.THIRD_CORE, geometry.BoundaryType.PERIODIC
        )
        quarterCartesian = geometry.SymmetryType(
            geometry.DomainType.QUARTER_CORE, geometry.BoundaryType.REFLECTIVE
        )

        self.assertTrue(geometry.checkValidGeomSymmetryCombo(geomHex, thirdPeriodic))
        self.assertTrue(geometry.checkValidGeomSymmetryCombo(geomHex, fullCore))
        self.assertTrue(
            geometry.checkValidGeomSymmetryCombo(geomCart, quarterCartesian)
        )
        self.assertTrue(geometry.checkValidGeomSymmetryCombo(geomRZT, quarterCartesian))
        self.assertTrue(geometry.checkValidGeomSymmetryCombo(geomRZ, fullCore))

        with self.assertRaises(ValueError):
            _ = geometry.SymmetryType(
                geometry.DomainType.THIRD_CORE,
                geometry.BoundaryType.REFLECTIVE,
                False,
            )
        with self.assertRaises(ValueError):
            geometry.checkValidGeomSymmetryCombo(geomHex, quarterCartesian)

        with self.assertRaises(ValueError):
            geometry.checkValidGeomSymmetryCombo(geomCart, thirdPeriodic)


if __name__ == "__main__":
    unittest.main()
