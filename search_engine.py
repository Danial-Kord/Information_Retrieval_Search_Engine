
from tokenizer import Tokenizer


tokens = set

inverted_index = dict

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
            elif ID == inverted_token_indexes[index]:
                return
        inverted_index[token].insert(place,ID)


def add_new_file(path,file_ID):
    with open(path, "r") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            # statement = "S " + i
            current_tokens = Tokenizer.get_tokens(line)
            for token in current_tokens:
                tokens.add(token)
                update_inverted_index_list(token,file_ID)







def main():
    test = "hi?he-----------llo, +-123****,bye///\\\\\\\didi^hmm"

    out = Tokenizer.get_tokens(test)

    for i in out:
        print(i)





if __name__ == "__main__":
    main()