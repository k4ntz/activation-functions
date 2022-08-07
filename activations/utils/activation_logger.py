import logging 

#https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
class ColoredFormatter(logging.Formatter):
    reset = '\x1b[0m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    white = '\x1b[38;5;231m'

    def __init__(self):
        logging.Formatter.__init__(self)
        format = "%(asctime)s - %(name)s - %(message)s (%(filename)s:%(lineno)d)"
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


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


class ActivationLogger(logging.Logger):
    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)

        colored_format = ColoredFormatter()
        console = logging.StreamHandler()
        console.setFormatter(colored_format)
        self.addHandler(console)
        self.filterDupl = DuplicateFilter()
        self.tracking_history = False

    def _track_history(self, want_track):
        if want_track:
            self.addFilter(self.filterDupl)
        else: 
            self.removeFilter(self.filterDupl)
    




logging.setLoggerClass(ActivationLogger)
