import codecs
import re


verbs_stemming = {}

split_characters = "[ ؟ ? . , : \" - - ؛ ; !#${}() % + = \\\  _ * ` ~ @ ^ / > < ، \n « »]+"

end_token_redundant = ['تر',
                    'ترین'
                    ,'ها'
                    ,'های'
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


    def make_verb_to_heritage_dict(verb_heritages_path):
        try:
            with codecs.open(verb_heritages_path, 'r', encoding='utf8') as f:
                Lines = f.readlines()
                for verb in Lines:
                    # print(verb)
                    Tokenizer.update_verb_to_heritage_dict(str(verb[0:len(verb)-3]))#removing ن at the end of the verb
                return True
        except:
            return False
    def update_verb_to_heritage_dict(verb):
        verbs_stemming[verb] = verb
        verb_splits = verb.split(" ")
        if len(verb_splits) > 1:
            # print("why")
            verb_e = verb_splits[len(verb_splits)-1]
            verb_s = ""
            for i in range(len(verb_splits)-1):
                verb_s+=verb_splits[i] + "‌"
            for start in start_verb_letters:
                for end in end_verb_letters:
                    new_verb = str(verb_s + start + verb_e + end)
                    verbs_stemming[new_verb] = verb
                    # print(new_verb)
        else:
            for start in start_verb_letters:
                for end in end_verb_letters:
                    new_verb = str(start + verb + end)
                    verbs_stemming[new_verb] = verb
                    # print(new_verb)



    def get_tokens(text):
        result = re.split(split_characters,text)
        output = []
        for i in result:
            if i.isnumeric():
                continue
            output.append(i)
        return output

    def get_normalized_tokens(text):
        outputSet = set()
        tokens = Tokenizer.get_tokens(text)
        for index in range(len(tokens)):
            normal = Tokenizer.token_normalizer(tokens[index])
            tokens[index] = normal
            outputSet.add(normal)
        return outputSet

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