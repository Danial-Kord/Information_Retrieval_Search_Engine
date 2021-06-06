from tokenizer import Tokenizer
import pandas as pd
import copy

min_high_frequency_tokens = 0.86

tokens = []

inverted_index = {}

ignorance_tokens =[]

urls = []

import re
# test re
# s = "Example String"
# replaced = re.sub('\\bE.*e\\b', 'a', s)
#
# p = re.compile(r'he( [^}]* )o',re.VERBOSE)
# p = p.sub(r'\1','hello')
# print(p)
#
# s = "سلام آنها میروند"
# p = re.compile('\\bمی'
#                   '(.*)'
#                   'ند\\b'
#                ,re.VERBOSE)
# replaced = p.sub(r'\1',s)
#
# print(replaced)


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


def add_new_file(path,content_ID_col_name,content_col_name,url_col_name):
    data = pd.read_excel(path)
    for index, row in data.iterrows():
        ID = row[content_ID_col_name]
        line = row[content_col_name]
        urls.append(row[url_col_name])
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



def answer_query(query):
    exported_tokens = Tokenizer.get_normalized_tokens(query)
    results_by_order = []
    current_tokens_index = []
    founded_inverted_index = []
    for i in range(len(exported_tokens)):
        results_by_order.append([])
        current_tokens_index.append(0)
        if inverted_index[exported_tokens[i]] is not None:
            founded_inverted_index.append(inverted_index[exported_tokens[i]])
        else:
            founded_inverted_index.append([])

    min_doc_ID = -1
    selected_doc_frequency = 0
    while(True):

        for index in range(len(exported_tokens)):
            token_doc_index = current_tokens_index[index]
            if len(founded_inverted_index[index]) > token_doc_index:
                if min_doc_ID == -1 or min_doc_ID > founded_inverted_index[index][token_doc_index]:
                    min_doc_ID = founded_inverted_index[index][token_doc_index]
                    selected_doc_frequency = 1
                elif min_doc_ID == founded_inverted_index[index][token_doc_index]:
                    selected_doc_frequency += 1
        if min_doc_ID == -1:
            break

        results_by_order[selected_doc_frequency-1].append(min_doc_ID)
        for index in range(len(exported_tokens)):
            token_doc_index = current_tokens_index[index]
            if len(founded_inverted_index[index]) > token_doc_index:
                if min_doc_ID == founded_inverted_index[index][token_doc_index]:
                    current_tokens_index[index]+=1
        min_doc_ID = -1;
        selected_doc_frequency = 0

    return results_by_order





def main():
    # test = "hi?he-----------llo, +-123****,bye///\\\\\\\didi^hmm"
    #
    # out = Tokenizer.get_tokens(test)
    #
    # for i in out:
    #     print(i)

    # test = "بدترین"
    # print(Tokenizer.token_normalizer(test))
    Tokenizer.make_verb_to_heritage_dict()
    # input()

    add_new_file("IR_Spring2021_ph12_7k.xlsx","id","content","url")
    clear_most_repeated_tokens()

    for i in tokens:
        print(i)

    while(True):
        query = input("query: ")
        print("Results:")
        index = 0
        for answer in answer_query(query):
            print(index+1)
            text = ""
            for i in answer:
                text += str(i) +" "
                print(urls[i-1])
            print(text)




if __name__ == "__main__":
    main()