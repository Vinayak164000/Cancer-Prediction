import sys

def error_msg_detail(error, error_detail: sys):
    _,_,exec_tn = error_detail.exc_info()
    file_name = exec_tn.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name,exec_tn.tb_lineno,str(error))

    return error_message

class CustomExceptions(Exception):
    def __init__(self, error_msg,error_detail: sys):
        super().__init__(error_msg)
        self.error_message = error_msg_detail(error_msg, error_detail = error_detail)

    def __str__(self):
        return self.error_message
    
