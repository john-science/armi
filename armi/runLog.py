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

r"""
This module handles logging of console output (e.g. warnings, information, errors)
during an armi run.

The default way of using the ARMI runLog is:

.. code::

    import armi.runLog as runLog
    runLog.setVerbosity('debug')
    runLog.info('information here')
    runLog.error('extra error info here')
    raise SomeException  # runLog.error() implies that the code will crash!

.. note::
    We plan to reimplement this with the standard Python logging module. It was customized
    to add a few features in a HPC/MPI environment but we now think we can use the standard
    system.

"""
from __future__ import print_function
import collections

import logging
import operator
import os
import sys
import time

from armi import context
from armi.context import Mode
from armi import meta

# define some private, module-level constants
_stderrName = "{0}.{1:04d}.stderr"
_stdoutName = "{0}.{1:04d}.stdout"

# NOTE: To reduce repeated calls to context later, there is a bit of ugly global
# execution here to define what the prefixes will be in every log line.
# NOTE: use ordereddict so we can get right order of options in GUI
_rank = ""
if context.MPI_RANK > 0:
    _rank = "-{:>03d}".format(context.MPI_RANK)
_logLevels = collections.OrderedDict(
    [
        ("debug", (0, "[dbug{}] ".format(_rank))),
        ("extra", (10, "[xtra{}] ".format(_rank))),
        ("info", (20, "[info{}] ".format(_rank))),
        ("important", (25, "[impt{}] ".format(_rank))),
        ("prompt", (27, "[prmt{}] ".format(_rank))),
        ("warning", (30, "[warn{}] ".format(_rank))),
        ("error", (50, "[err {}] ".format(_rank))),
        ("header", (100, "".format(_rank))),
    ]
)
_whitespace = " " * len(max([l[1] for l in _logLevels.values()]))

# modify the logging module strings for printing
for logValue, shortLogString in _logLevels.values():
    logging.addLevelName(logValue, shortLogString)


def getLogVerbosityLevels():
    """Return a list of the available log levels (e.g., debug, extra, etc.)."""
    return list(_logLevels.keys())


def getLogVerbosityRank(level):
    """Return integer verbosity rank given the string verbosity name."""
    if level not in getLogVerbosityLevels():
        raise KeyError(
            "{} is not a valid verbosity level. Choose from {}".format(
                level, getLogVerbosityLevels()
            )
        )
    return _logLevels[level][0]


def _checkLogVerbsityRank(rank):
    """Check that the verbosity rank is defined within the _logLevels and return it if it is."""
    validRanks = []
    for level in getLogVerbosityLevels():
        expectedRank = getLogVerbosityRank(level)
        validRanks.append(rank)
        if rank == expectedRank:
            return rank
    raise KeyError(
        "Invalid verbosity rank {}. Valid options are: {}".format(rank, validRanks)
    )


class Log:
    """
    Abstract class that ARMI code calls to be rendered to some kind of log.
    """

    def __init__(self, mpiRank=0, mpiSize=1):
        """
        Build a log object

        Parameters
        ----------
        mpiRank : int
            If this is zero, we are in the parent process, otherwise child process.
            The default of 0 means we assume the parent process.
            This should not be adjusted after instantiation.
        mpiSize : int
            If this is one, we are in the parent process, with child process.
            This should not be adjusted after instantiation.

        """
        self._mpiRank = mpiRank
        self._mpiSize = mpiSize
        self._verbosity = logging.INFO
        self.name = "log"
        self._singleMessageCounts = collections.defaultdict(lambda: 0)
        self._singleWarningMessageCounts = collections.defaultdict(lambda: 0)
        self.loggerName = (
            "parent" if self._mpiRank == 0 else "{:>03d}".format(self._mpiRank)
        )
        self.logger = logging.getLogger(self.loggerName)
        self._errStream = None
        # https://docs.python.org/2/library/sys.html says
        # to explicitly save these instead of relying upon __stderr__, etc.
        if self._mpiRank == 0:
            self.initialErr = sys.stderr
        else:
            # Attach error stream to a null device until we open log files. We don't know what to
            # call them until we have processed some of the settings, and any errors encountered in
            # that should be encountered on the master process anyway.
            self.initialErr = open(os.devnull, "w")
        self.setErrStream(self.initialErr)

        # set up the parent logger
        if self._mpiRank == 0:
            logging.basicConfig(
                level=self._verbosity, format="%(levelname)s%(message)s"
            )

    def flush(self):
        self._errStream.flush()

    def standardLogMsg(self, msgType, msg, single=False, label=None):
        """
        Add formatting to a message and handle its singleness, if applicable.

        This is a wrapper around logger.log() that does most of the work and is
        used by all message passers (e.g. info, warning, etc.).
        """
        # the message label is only used to determine unique for single-print warnings
        if label is None:
            label = msg

        # Skip writing the message if it is below the set verbosity
        msgVerbosity, msgLabel = _logLevels[msgType]
        if msgVerbosity < self._verbosity:
            return

        # Skip writing the message if it is single-print warning
        if single and self._msgHasAlreadyBeenEmitted(label, msgType):
            return

        # Do the actual logging, but add that custom indenting first
        msg = self._cleanMsg(msg)
        self.logger.log(msgVerbosity, msg)

    def _cleanMsg(self, msg):
        """Messages need to be strings, and tabbed if multi-line"""
        return str(msg).rstrip().replace("\n", "\n" + _whitespace)

    def _msgHasAlreadyBeenEmitted(self, label, msgType=""):
        """Return True if the count of the label is greater than 1."""
        if msgType == "warning" or msgType == "critical":
            self._singleWarningMessageCounts[label] += 1
            if (
                self._singleWarningMessageCounts[label] > 1
            ):  # short circuit because error has changed
                return True
        else:
            self._singleMessageCounts[label] += 1
            if (
                self._singleMessageCounts[label] > 1
            ):  # short circuit because error has changed
                return True
        return False

    def clearSingleWarnings(self):
        """Reset the single warned list so we get messages again."""
        self._singleMessageCounts.clear()

    def warningReport(self):
        """Summarize all warnings for the run."""
        info("----- Final Warning Count --------")
        info("  {0:^10s}   {1:^25s}".format("COUNT", "LABEL"))

        # sort by labcollections.defaultdict(lambda: 1)
        for label, count in sorted(
            self._singleWarningMessageCounts.items(), key=operator.itemgetter(1)
        ):
            info("  {0:10d}   {1:<25s}".format(count, str(label)))
        info("------------------------------------")

    def setVerbosity(self, levelInput):
        """
        Sets the minimum output verbosity for the logger.

        Any message with a higher verbosity than this will
        be emitted.

        Parameters
        ----------
        levelInput : int or str
            The level to set the log output verbosity to.
            Valid numbers are 0-50 and valid strings are keys of _logLevels

        Examples
        --------
        >>> setVerbosity('debug') -> sets to 0
        >>> setVerbosity(0) -> sets to 0

        """
        self._verbosity = (
            getLogVerbosityRank(levelInput)
            if isinstance(levelInput, str)
            else _checkLogVerbsityRank(levelInput)
        )

    def getVerbosity(self):
        """Return the global runLog verbosity."""
        return self._verbosity

    def setErrStream(self, stderr):
        """Set the stderr stream to any stream object."""
        sys.stderr = self._errStream = stderr

    def _getErrStream(self):
        return self._errStream

    def _restoreErrStream(self):
        """Set the system stderr to their defaults (as they were when the run started)."""
        if sys.stderr == self._errStream:
            sys.stderr = self.initialErr  # sys.__stderr__


class PrintLog(Log):
    """Log that emits to stdout/stderr or file-based streams (for MPI workers) with print."""

    def startLog(self, name):
        """Initialize the streams when parallel processing"""
        self.name = os.path.join("logs", name)

        # create log dir
        if self._mpiRank == 0:
            self._createLogDir()

        # set up the child loggers
        if self._mpiRank > 0:
            self._verbosity = logging.WARNING
            logging.basicConfig(
                filename=os.path.join(
                    self.name, _stdoutName.format(self.name, self._mpiRank)
                ),
                level=self._verbosity,
                format="%(levelname)s%(message)s",
            )

        # don't worry about MPI blocking if this isn't a parallel run
        if self._mpiSize == 1 or context.MPI_COMM is None:
            return
        context.MPI_COMM.barrier()

        # init stderr file, if you are in a child process
        if self._mpiRank > 0:
            errStream = open(_stderrName.format(self.name, self._mpiRank), "w")
            self.setErrStream(errStream)

    def _createLogDir(self):
        """A helper method to create the log directory"""
        # make the directory
        if not os.path.exists("logs"):
            try:
                os.makedirs("logs")
            except FileExistsError:
                pass

        # stall until it shows up in file system (SMB caching issue?)
        while not os.path.exists("logs"):
            time.sleep(0.1)

    def close(self):
        """End use of the log. Concatenate if needed and restore defaults"""
        if self._mpiRank == 0 and self._mpiSize > 1:
            try:
                self.concatenateLogs()
            except IOError as ee:
                warning("Failed to concatenate logs due to IOError.")
                error(ee)
        elif self._mpiRank > 0 and self._mpiSize > 1:
            self._errStream.flush()
            self._errStream.close()
        self._restoreErrStream()

    def concatenateLogs(self):
        """
        Concatenate the armi run logs and delete them.

        Should only ever be called by parent.
        """
        info("Concatenating {0} standard streams".format(self._mpiSize))
        for rank in range(1, self._mpiSize):
            # first, print the log messages for a child process
            stdoutName = os.path.join("logs", _stdoutName.format(self.name, rank))
            if os.path.exists(stdoutName):
                with open(stdoutName, "r") as logFile:
                    data = logFile.read()
                    if data:
                        # only write if there's something to write.
                        rankId = "\n{0} RANK {1:03d} STDOUT {2}\n".format(
                            "-" * 10, rank, "-" * 60
                        )
                        print(rankId)
                        print(data)
                try:
                    os.remove(stdoutName)
                except OSError:
                    warning("Could not delete {0}".format(stdoutName))

            # then print the stderr messages for that child process
            stderrName = os.path.join("logs", _stderrName.format(self.name, rank))
            if os.path.exists(stdoutName):
                with open(stderrName) as logFile:
                    data = logFile.read()
                    if data:
                        # only write if there's something to write.
                        rankId = "\n{0} RANK {1:03d} STDERR {2}\n".format(
                            "-" * 10, rank, "-" * 60
                        )
                        print(rankId, file=sys.stderr)
                        print(data, file=sys.stderr)
                try:
                    os.remove(stderrName)
                except OSError:
                    warning("Could not delete {0}".format(stderrName))


# Here are all the module-level functions that should be used for most outputs.
# They use the PrintLog object behind the scenes.
def raw(msg):
    """
    Print raw text without any special functionality.
    """
    LOG.standardLogMsg("header", msg, single=False, label=msg)


def extra(msg, single=False, label=None):
    LOG.standardLogMsg("extra", msg, single=single, label=label)


def debug(msg, single=False, label=None):
    LOG.standardLogMsg("debug", msg, single=single, label=label)


def info(msg, single=False, label=None):
    LOG.standardLogMsg("info", msg, single=single, label=label)


def important(msg, single=False, label=None):
    LOG.standardLogMsg("important", msg, single=single, label=label)


def warning(msg, single=False, label=None):
    LOG.standardLogMsg("warning", msg, single=single, label=label)


def error(msg, single=False, label=None):
    LOG.standardLogMsg("error", msg, single=single, label=label)


def header(msg, single=False, label=None):
    LOG.standardLogMsg("header", msg, single=single, label=label)


def flush():
    """Flush LOG's output in the stderr stream"""
    LOG.flush()


def prompt(statement, question, *options):
    """Prompt the user for some information."""
    from armi.localization import exceptions

    if context.CURRENT_MODE == Mode.GUI:
        # avoid hard dependency on wx
        import wx  # pylint: disable=import-error

        msg = statement + "\n\n\n" + question
        if len(msg) < 300:
            style = wx.CENTER
            for opt in options:
                style |= getattr(wx, opt)
            dlg = wx.MessageDialog(None, msg, style=style)
        else:
            # for shame. Might make sense to move the styles stuff back into the
            # Framework
            from tparmi.gui.styles import dialogues

            dlg = dialogues.ScrolledMessageDialog(None, msg, "Prompt")
        response = dlg.ShowModal()
        dlg.Destroy()
        if response == wx.ID_CANCEL:
            raise exceptions.RunLogPromptCancel("Manual cancellation of GUI prompt")
        return response in [wx.ID_OK, wx.ID_YES]

    elif context.CURRENT_MODE == Mode.INTERACTIVE:
        response = ""
        responses = [
            opt for opt in options if opt in ["YES_NO", "YES", "NO", "CANCEL", "OK"]
        ]

        if "YES_NO" in responses:
            index = responses.index("YES_NO")
            responses[index] = "NO"
            responses.insert(index, "YES")

        if not any(responses):
            raise RuntimeError("No suitable responses in {}".format(responses))

        # highly requested shorthand responses
        if "YES" in responses:
            responses.append("Y")
        if "NO" in responses:
            responses.append("N")

        while response not in responses:
            LOG.standardLogMsg("prompt", statement)
            LOG.standardLogMsg(
                "prompt", "{} ({}): ".format(question, ", ".join(responses))
            )
            response = sys.stdin.readline().strip().upper()

        if response == "CANCEL":
            raise exceptions.RunLogPromptCancel(
                "Manual cancellation of interactive prompt"
            )

        return response in ["YES", "Y", "OK"]

    else:
        raise exceptions.RunLogPromptUnresolvable(
            "Incorrect CURRENT_MODE for prompting user: {}".format(context.CURRENT_MODE)
        )


def warningReport():
    LOG.warningReport()


def setVerbosity(level):
    # convenience function
    LOG.setVerbosity(level)


def getVerbosity():
    return LOG.getVerbosity()


# ---------------------------------------


def logFactory():
    """Create the default logging object."""
    return PrintLog(int(context.MPI_RANK), int(context.MPI_SIZE))


LOG = logFactory()
