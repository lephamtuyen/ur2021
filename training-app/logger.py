import os
import sys
import tempfile
import datetime

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    """
    Key Value writer
    """

    def writekvs(self):
        """
        write a dictionary to file

        :param kvs: (dict)
        """
        raise NotImplementedError


class SeqWriter(object):
    """
    sequence writer
    """

    def writeseq(self, seq):
        """
        write an array to file

        :param seq: (list)
        """
        raise NotImplementedError


class HumanOutputFormat(SeqWriter):
    def __init__(self, filename_or_file):
        """
        log to a file, in a human readable format

        :param filename_or_file: (str or File) the file to write the log to
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            assert hasattr(filename_or_file, 'write'), 'Expected file or str, got {}'.format(filename_or_file)
            self.file = filename_or_file
            self.own_file = False

    @classmethod
    def _truncate(cls, string):
        return string[:20] + '...' if len(string) > 23 else string

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(' ')
        self.file.write('\n')
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        """
        log to a file, in a CSV format

        :param filename: (str) the file to write the log to
        """
        self.file_name = filename
        self.file = open(filename, 'w+t')
        self.name2val = {}
        self.keys = []
        self.sep = ','

    def writekvs(self):
        if (len(self.name2val) == 0):
            return

        # Add our current row to the history
        extra_keys = self.name2val.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, key) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(key)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            value = self.name2val.get(key)
            if value is not None:
                self.file.write(str(value))
        self.file.write('\n')
        self.file.flush()
        self.name2val.clear()

    def close(self):
        """
        closes the file
        """
        self.file.close()


def make_file(file_name, ev_dir):
    """
    return a logger for the requested format

    :param file_name: (str) the requested format to log to ('stdout', 'log', 'csv')
    :param ev_dir: (str) the logging directory
    :return: (KVWrite) the logger
    """
    os.makedirs(ev_dir, exist_ok=True)
    if 'stdout' in file_name:
        return HumanOutputFormat(sys.stdout)
    elif '.txt' in file_name:
        return HumanOutputFormat(os.path.join(ev_dir, file_name))
    elif '.csv' in file_name:
        return CSVOutputFormat(os.path.join(ev_dir, file_name))
    else:
        raise ValueError('Unknown format specified: %s' % (file_name,))


# ================================================================
# API
# ================================================================

def logkv(key, val, log_file):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.

    :param key: (Any) save to log this key
    :param val: (Any) save to log this value
    :param log_file: (Any) save to log_file
    """
    Logger.CURRENT.logkv(key, val, log_file)


def logkvs(key_values, log_file):
    """
    Log a dictionary of key-value pairs

    :param key_values: (dict) the list of keys and values to save to log
    :param log_file: (Any) save to log_file
    """
    for key, value in key_values.items():
        logkv(key, value, log_file)


def dumpkvs(log_file=None):
    """
    Write all of the diagnostics from the current iteration
    """
    Logger.CURRENT.dumpkvs(log_file)


def getkvs():
    """
    get the key values logs

    :return: (dict) the logged values
    """
    return Logger.CURRENT.name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.

    :param args: (list) log the arguments
    :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    Logger.CURRENT.log(*args, level=level)


def debug(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the DEBUG level.

    :param args: (list) log the arguments
    """
    log(*args, level=DEBUG)


def info(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the INFO level.

    :param args: (list) log the arguments
    """
    log(*args, level=INFO)


def warn(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the WARN level.

    :param args: (list) log the arguments
    """
    log(*args, level=WARN)


def error(*args):
    """
    Write the sequence of args, with no separators,
    to the console and output files (if you've configured an output file).
    Using the ERROR level.

    :param args: (list) log the arguments
    """
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.

    :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    Logger.CURRENT.set_level(level)


def get_level():
    """
    Get logging threshold on current logger.
    :return: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
    """
    return Logger.CURRENT.level


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)

    :return: (str) the logging directory
    """
    return Logger.CURRENT.get_dir()


# ================================================================
# Backend
# ================================================================

class Logger(object):
    CURRENT = None

    def __init__(self, folder, output_files):
        """
        the logger class

        :param folder: (str) the logging location
        :param output_files: ([str]) the list of output file
        """
        self.level = INFO
        self.dir = folder
        self.output_files = output_files

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val, log_file):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: (Any) save to log this key
        :param val: (Any) save to log this value
        """
        self.output_files[log_file].name2val[key] = val

    def dumpkvs(self, log_file):
        """
        Write all of the diagnostics from the current iteration
        """
        if self.level == DISABLED:
            return

        if log_file is not None:
            self.output_files[log_file].writekvs()
        else:
            for _, fmt in self.output_files.items():
                if isinstance(fmt, KVWriter):
                    fmt.writekvs()

    def log(self, *args, level=INFO):
        """
        Write the sequence of args, with no separators,
        to the console and output files (if you've configured an output file).

        level: int. (see logger.py docs) If the global logger level is higher than
                    the level argument here, don't print to stdout.

        :param args: (list) log the arguments
        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        """
        Set logging threshold on current logger.

        :param level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        """
        self.level = level

    def get_dir(self):
        """
        Get directory that log files are being written to.
        will be None if there is no output directory (i.e., if you didn't call start)

        :return: (str) the logging directory
        """
        return self.dir

    def close(self):
        """
        closes the file
        """
        for _, fmt in self.output_files.items():
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        """
        log to the requested format outputs

        :param args: (list) the arguments to log
        """
        for _, fmt in self.output_files.items():
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


# sook 20190603
# Yes.. I was looking for this. I mean this `logger.py` script is called from `data.py` and
# `data.py` uses log() function inside itself, which calls this `Logger.CURRENT` in turn.
# I was wondering where this `Logger.CURRENT` is defined or initialized!

# O2 reference point
# O2: So I'm moving this to the very bottom of the script for better visibility.

# Logger.CURRENT = Logger(folder=None, output_files=[HumanOutputFormat(sys.stdout)])

def configure(folder=None, log_files=None):
    """
    configure the current logger

    :param folder: (str) the save location
    :param log_files: (list) the output logging
    """
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(),
                              datetime.datetime.now().strftime("bakingsoda-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    if log_files is None:
        log_files = ['stdout', 'log.txt']
    else:
        log_files = ['stdout', 'log.txt'] + log_files

    output_files = {}
    for file in log_files:
        output_files[file] = make_file(file, folder)

    Logger.CURRENT = Logger(folder=folder, output_files=output_files)
    log('Logging to %s' % folder)


# ================================================================
# Readers
# ================================================================

def read_csv(fname):
    """
    read a csv file using pandas

    :param fname: (str) the file path to read
    :return: (pandas DataFrame) the data in the csv
    """
    import pandas
    return pandas.read_csv(fname, index_col=None, comment='#')


Logger.CURRENT = Logger(folder=None, output_files=[HumanOutputFormat(sys.stdout)])
