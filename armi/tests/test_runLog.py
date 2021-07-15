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

import logging
import os
import pytest
import sys
import unittest

from armi import runLog


class TestRunLog(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        """This pytest fixture allows us to caption logging messages that pytest interupts"""
        self._caplog = caplog

    def test_setVerbosityFromInteger(self):
        """Test that the log verbosity can be set with an integer."""
        expectedStrVerbosity = runLog.getLogVerbosityLevels()[0]
        verbosityRank = runLog.getLogVerbosityRank(expectedStrVerbosity)
        runLog.setVerbosity(verbosityRank)
        self.assertEqual(verbosityRank, runLog.getVerbosity())

    def test_setVerbosityFromString(self):
        """Test that the log verbosity can be set with a string."""
        expectedStrVerbosity = runLog.getLogVerbosityLevels()[0]
        verbosityRank = runLog.getLogVerbosityRank(expectedStrVerbosity)
        runLog.setVerbosity(expectedStrVerbosity)
        self.assertEqual(verbosityRank, runLog.getVerbosity())

    def test_invalidSetVerbosityByRank(self):
        """Test that the log verbosity setting fails if the integer is invalid."""
        with self.assertRaises(KeyError):
            runLog.setVerbosity(5000)

    def test_invalidSetVerbosityByString(self):
        """Test that the log verbosity setting fails if the integer is invalid."""
        with self.assertRaises(KeyError):
            runLog.setVerbosity("taco")

    def test_caplogBasicParentRunLog(self):
        """A basic test of the logging of the parent runLog"""
        with self._caplog.at_level(logging.INFO):
            runLog.debug("This is invisible.")
            runLog.info("one")
            runLog.important("two two\n")
            runLog.warning("three three three")
            runLog.error("four four four four")

        messages = [r.message for r in self._caplog.records]
        assert len(messages) > 0
        assert "one" in messages[0]
        assert "two two" in messages[1]
        assert "three three three" in messages[2]
        assert "four four four four" in messages[3]

    def test_caplogBasicChildRunLog(self):
        """A basic test of the logging of the child runLog"""
        with self._caplog.at_level(logging.WARNING):
            pl = runLog.PrintLog(1, 2)
            pl._createLogDir()
            pl.startLog("test_caplogBasicChildRunLog")
            pl.standardLogMsg(
                "debug", "You shouldn't see this.", single=False, label=None
            )
            pl.standardLogMsg("warning", "Hello, ", single=False, label=None)
            pl.standardLogMsg("error", "world!", single=False, label=None)

        messages = [r.message for r in self._caplog.records]
        assert len(messages) > 0
        assert "Hello" in messages[0]
        assert "world" in messages[1]


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
