from tokenizer import Tokenizer
import pandas as pd
import copy

min_high_frequency_tokens = 0.8

tokens = []

inverted_index = {}

ignorance_tokens =[]

def update_inverted_index_list(token,ID):
    if inverted_index.keys().__contains__(token) is False:
        inverted_index[token] = []
        inverted_index[token].append(ID)
    else:
        inverted_token_indexes = inverted_index[token]
        place = 0
        for index in range(len(inverted_token_indexes)):
            if ID > inverted_token_indexes[index]:
                place += 1
            elif ID < inverted_token_indexes[index]:
                break
            else:
                return
        inverted_index[token].insert(place,ID)


def add_new_file(path,content_ID_col_name,content_col_name):
    data = pd.read_excel(path)
    for index, row in data.iterrows():
        ID = row[content_ID_col_name]
        line = row[content_col_name]
        # statement = "S " + i
        print(ID)
        current_tokens = Tokenizer.get_tokens(line)
        for token in current_tokens:
            if token in ignorance_tokens:
                continue
            normalized_token = Tokenizer.token_normalizer(token)
            if not normalized_token in tokens:
                tokens.append(normalized_token)
            update_inverted_index_list(normalized_token,ID)






def clear_most_repeated_tokens():
    copy_tokens = copy.deepcopy(tokens)
    for token in copy_tokens:
        if len(inverted_index[token]) >= len(copy_tokens) * min_high_frequency_tokens:
            tokens.remove(token)
            inverted_index.pop(token)
    copy_tokens.clear()





def main():
    test = "hi?he-----------llo, +-123****,bye///\\\\\\\didi^hmm"

    out = Tokenizer.get_tokens(test)

    for i in out:
        print(i)

    test = "بدترین"
    print(Tokenizer.token_normalizer(test))
    input()

    add_new_file("C:\Danial\Projects\Danial\Information_Retrieval_Search_Engine\IR_Spring2021_ph12_7k.xlsx","id","content")
    clear_most_repeated_tokens()

    for i in tokens:
        print(i)





if __name__ == "__main__":
    main()