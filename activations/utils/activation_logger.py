import logging 
import os

#https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
class ColoredFormatter(logging.Formatter):
    reset = '\x1b[0m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    white = '\x1b[38;5;231m'

    def __init__(self, format):
        logging.Formatter.__init__(self, format)
        self.fmt = format

        self.FORMATS = {
            logging.DEBUG: self.blue + self.fmt + self.reset, 
            logging.INFO: self.white + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset, 
            logging.ERROR: self.red + self.fmt + self.reset, 
            logging.WARNING: self.yellow + self.fmt + self.reset
        }

    


    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ActivationLogger(object):

    def __init__(self, logger_name, log_level = logging.DEBUG, show_logger_name = True, show_time = False, file = "tst"):
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(log_level)
        console = logging.StreamHandler()
        console.setLevel(log_level)
        self.logger_history = None
        self.show_name = show_logger_name
        self.show_time = show_time



        messageFormat = self.getFormatter("tst")
        self.console = console
        if os.name != 'nt':
            console.setFormatter(ColoredFormatter(messageFormat))
        if os.name == 'nt':
            console.setFormatter(messageFormat)

        #TODO: is this relevant?
        #if not self._logger.hasHandlers():
        self._logger.addHandler(self.console)


    def setFormatter(self, filename):
        format = self.getFormatter(filename)
        c_fmt = ColoredFormatter(format)
        self.console.setFormatter(c_fmt)
        

    def _track_history(self, save = True):
        if save:
            self.logger_history = set([])
        else: 
            self.logger_history = None

    def log_multiple(self, msg, func):
        if self.logger_history is not None: 
            if msg not in self.logger_history:
                func(msg)
                self.logger_history.add(msg)
        else: 
            func(msg)


    def debug(self, msg):
        func = self._logger.debug
        self.log_multiple(msg, func)

    def warn(self, msg):
        func = self._logger.warn
        self.log_multiple(msg, func)

    def info(self, msg):
        func = self._logger.info
        self.log_multiple(msg, func)

    def error(self, msg):
        func = self._logger.error
        self.log_multiple(msg, func)

    def critical(self, msg):
        func = self._logger.critical
        self.log_multiple(msg, func)

    def getFormatter(self, filename_set):
        format = ''

        if self.show_name:
            format = '%(name)s'

        if self.show_time:
            if len(format) > 0:
                format = format + ' | '
            
            format = format + '%(asctime)s'

        if len(format) > 0:
            format = format + ' | '
        format = format + '%(filename_set)s | %(message)s'
        return format