import codecs
import re

verb_heritages_path = "C:\Danial\Projects\Danial\Information_Retrieval_Search_Engine\\verbs.txt"
verbs_stemming = {}

split_characters = "[?., : ; !#${}() % + = \\\  _ * ` ~ @ ^ / > < ، \n « »]+"

end_token_redundant = ['تر',
                    'ترین'
                    ,'ها'
                    ,'اً'
                    ,'گاه'
                    ,'ی'
                    ,'ات'
                    ]

start_token_redundant = ['هم'
                         ]

end_verb_letters = [
    'م',
    'ی',
    '',
    'یم',
    'ید',
    'ند',
    'ه'

]

start_verb_letters = [
    'می‌',
    '',
    'نمی‌',
    'ن',
]

class Tokenizer:

    def __init__(self):
        print("tokenizer initiated")


    def make_verb_to_heritage_dict():
        with codecs.open(verb_heritages_path, 'r', encoding='utf8') as f:
            Lines = f.readlines()
            for verb in Lines:
                print(verb)
                Tokenizer.update_verb_to_heritage_dict(str(verb[0:len(verb)-3]))#removing ن at the end of the verb

    def update_verb_to_heritage_dict(verb):
        verbs_stemming[verb] = verb
        verb_splits = verb.split(" ")
        if len(verb_splits) > 1:
            print("why")
            verb_e = verb_splits[len(verb_splits)-1]
            verb_s = ""
            for i in range(len(verb_splits)-1):
                verb_s+=verb_splits[i] + "‌"
            for start in start_verb_letters:
                for end in end_verb_letters:
                    new_verb = str(verb_s + start + verb_e + end)
                    verbs_stemming[new_verb] = verb
                    print(new_verb)
        else:
            for start in start_verb_letters:
                for end in end_verb_letters:
                    new_verb = str(start + verb + end)
                    verbs_stemming[new_verb] = verb
                    print(new_verb)



    def get_tokens(text):
        return re.split(split_characters,text)

    def get_normalized_tokens(text):
        tokens = Tokenizer.get_tokens(text)
        for index in range(len(tokens)):
            tokens[index] = Tokenizer.token_normalizer(tokens[index])
        return tokens

    def token_normalizer(token):
        if verbs_stemming.keys().__contains__(token):
            return verbs_stemming[token]
        for i in end_token_redundant:
            if token.endswith(i):
                token = token[0:len(token) - len(i)]
                break
        for i in start_token_redundant:
            if token.startswith(i):
                token = token[len(i):len(token)]
                break
        return token