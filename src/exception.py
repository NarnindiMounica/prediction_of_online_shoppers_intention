import sys
from src.logger import logging

def error_message_details(error, error_details):
    #type(error), error, error_traceback = sys.exc_info()
    _, _, exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    linenum = exc_tb.tb_lineno
    error_message =' Error occurred from file {}, at line number {}, {}'.format(filename, linenum, error)

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details=error_details)

    def __str__(self):
        return self.error_message


        


