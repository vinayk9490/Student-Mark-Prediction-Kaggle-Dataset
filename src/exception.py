import sys
from venv import logger
from src.logger import logging

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message



class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

'''
in code we raise the exception something like this:
except Exception as e:
    raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        a = (1/0)
    except Exception as e:
        logging.info("Error message: {}".format(e))
        raise CustomException(e, sys)

Sys: For Exception Handling.
In the context of exception handling, the sys module is mainly used to get detailed information about errors and control how the program exits when an exception occurs.
'''