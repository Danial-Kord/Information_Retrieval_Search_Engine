
import re


split_characters = "[?., !#${}() % + = \\\ \\- _ * ` ~ @ ^ /]+"

class Tokenizer:

    def __init__(self):
        print("tokenizer initiated")


    def get_tokens(text):
        return re.split(split_characters,text)
