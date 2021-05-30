
import re


split_characters = "[?., : ; !#${}() % + = \\\  _ * ` ~ @ ^ / > < ، \n]+"

end_token_redundant = ['تر',
                    'ترین'
                    ,'ها'
                    ,'اً'
                    ,'گاه'
                    ,'ی'
                    ,'ات'
                    ]

start_token_redundant = ['می',
                         'هم'
                    ]

class Tokenizer:

    def __init__(self):
        print("tokenizer initiated")


    def get_tokens(text):
        return re.split(split_characters,text)

    def get_normalized_tokens(text):
        tokens = Tokenizer.get_tokens(text)
        for index in range(len(tokens)):
            tokens[index] = Tokenizer.token_normalizer(tokens[index])
        return tokens

    def token_normalizer(token):
        for i in end_token_redundant:
            if token.endswith(i):
                token = token[0:len(token) - len(i)]
                break
        for i in start_token_redundant:
            if token.startswith(i):
                token = token[len(i):len(token)]
                break
        return token