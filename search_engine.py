from tokenizer import Tokenizer
import pandas as pd
import copy
import math
from MinHeap import MinHeap
min_high_frequency_tokens = 0.86

tokens = []

inverted_index = {}
champion_list = {} #collection of the best from inverted index
r = 20 #max len of the champion list for every term

ignorance_tokens =[]

urls = {}

df = {} #documents frequency of words

#doc term frequency(doc ID are rows, each row has a dictionary of tokens to frequency)
tdf = {}
k = 5 #return the 5 best results
total_docs = 0
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


def tfIdf_calculator(token,doc_ID):
    if tdf[doc_ID].keys().__contains__(token) is False:
        return 0
    return tfIdf_calc(token,tdf[doc_ID][token])

def tfIdf_calc(term,term_frequency):
    return (1 + math.log(term_frequency)) * math.log(total_docs / df[term])

def make_championList():
    for term in tokens:
        index = 0
        min_index = 0
        min_value = tdf[inverted_index[term][0]][term]
        for i in inverted_index[term]:
            if index < r:
                champion_list[term].append(i)
                if min_value > tdf[i][term]:
                    min_value = i
                    min_index = index
                index+=1
            elif min_value < i:
                min_value = i
                champion_list[term].pop(min_index)
                champion_list[term].append(i)
                for j in range(len(champion_list[term])):
                    if tdf[champion_list[term][j]][term] < min_value:
                        min_index = j
                        min_value = champion_list[term][j]



def update_inverted_index_list(token,ID,increamental = False):
    if inverted_index.keys().__contains__(token) is False:
        tokens.append(token)
        champion_list[token] = []
        inverted_index[token] = []
        inverted_index[token].append(ID)
    elif increamental:
        if inverted_index[token][len(inverted_index[token]) -1 ] != ID:
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

#get text and updates the frequency of token
def update_term_frequency(normalizedToken,originalToken,text,doc_ID):

    if df.keys().__contains__(normalizedToken) is False:
        df[normalizedToken] = 0


    if tdf.keys().__contains__(doc_ID) is False:
        tdf[doc_ID] = {}
        global total_docs
        total_docs += 1

    if tdf[doc_ID].keys().__contains__(normalizedToken) is False:
        tdf[doc_ID][normalizedToken] = 0
        df[normalizedToken] += 1
    count = text.count(originalToken)
    tdf[doc_ID][normalizedToken] += count



def add_new_file(path,content_ID_col_name,content_col_name,url_col_name):
    data = pd.read_excel(path)
    for index, row in data.iterrows():
        ID = row[content_ID_col_name]
        line = row[content_col_name]
        urls[ID] = row[url_col_name]
        # statement = "S " + i
        # print(ID)

        current_tokens = Tokenizer.get_tokens(line)
        for token in current_tokens:
            if token in ignorance_tokens:
                continue
            normalized_token = Tokenizer.token_normalizer(token)
            update_inverted_index_list(normalized_token,ID,True)
            update_term_frequency(normalized_token,token,line,ID)

    make_championList()
    return data


# gives the similarity of 2 vectors
def cosin_sim_calculator(query_vector,doc_vector):
    similarity = 0
    first_vector_len = 0
    second_vector_len = 0
    for index in range(len(query_vector)):
        v1 = query_vector[index]
        v2 = doc_vector[index]
        similarity += v1 * v2
        first_vector_len += v1 ** 2
        second_vector_len += v2 ** 2
    return similarity / (math.sqrt(first_vector_len * second_vector_len))





def clear_most_repeated_tokens():
    copy_tokens = copy.deepcopy(tokens)
    for token in copy_tokens:
        if len(inverted_index[token]) >= len(copy_tokens) * min_high_frequency_tokens:
            tokens.remove(token)
            inverted_index.pop(token)
            ignorance_tokens.append(token)
    copy_tokens.clear()


def champion_tfidf_query(query):
    results = 0
    all_docs = set()
    exported_tokens_temp = Tokenizer.get_normalized_tokens(query)
    exported_tokens = []
    founded_inverted_index = []

    query_vector = []

    for i in exported_tokens_temp:
        if champion_list.keys().__contains__(i):
            print(i)
            exported_tokens.append(i)
            founded_inverted_index.append(champion_list[i])
            query_vector.append(tfIdf_calc(i, 1))

    for i in founded_inverted_index:
        for id in i:
            all_docs.add(id)

    doc_score = {}
    minheap = MinHeap(r)
    for i in all_docs:
        vector_result = []
        for term in exported_tokens:
            vector_result.append(tfIdf_calculator(term, i))
        doc_score[i] = cosin_sim_calculator(query_vector,vector_result)
        minheap.insert(-doc_score[i],i)
    minheap.minHeap()

    if len(all_docs) > k:
        return minheap





    query_vector = []

def query_handler(query,method = 0):
    if method == 0:
        return simple_query(query)
    if method == 1:
        return tfIdf_cosine_query(query)
    if method == 2:
        return champion_tfidf_query(query)

# calculates the vector space presentation and similarity and sort by heap(return the sorted MinHeap)
def tfIdf_cosine_query(query):
    exported_tokens_temp = Tokenizer.get_normalized_tokens(query)

    current_tokens_index = []
    founded_inverted_index = []
    query_vector = []
    exported_tokens = []
    for i in exported_tokens_temp:
        if inverted_index.keys().__contains__(i):
            exported_tokens.append(i)

    for i in range(len(exported_tokens)):
        print(exported_tokens[i])
        current_tokens_index.append(0)#current doc ID
        if inverted_index.keys().__contains__(exported_tokens[i]):
            founded_inverted_index.append(inverted_index[exported_tokens[i]])
            query_vector.append(tfIdf_calc(exported_tokens[i], 1))
        else:
            founded_inverted_index.append([])

    min_doc_ID = -1

    doc_score = {}

    minheap = MinHeap(100)




    while (True):
        for index in range(len(exported_tokens)):
            token_doc_index = current_tokens_index[index]
            if len(founded_inverted_index[index]) > token_doc_index:
                if min_doc_ID == -1 or min_doc_ID > founded_inverted_index[index][token_doc_index]:
                    min_doc_ID = founded_inverted_index[index][token_doc_index]

        if min_doc_ID == -1:
            break


        # calculates the vector space presentation of doc tfIdf for selected ID
        vector_result = []
        for index in range(len(exported_tokens)):
            token_doc_index = current_tokens_index[index]
            if len(founded_inverted_index[index]) > token_doc_index:
                if min_doc_ID == founded_inverted_index[index][token_doc_index]:
                    current_tokens_index[index] += 1
            vector_result.append(tfIdf_calculator(exported_tokens[index], min_doc_ID))

        doc_score[min_doc_ID] = cosin_sim_calculator(query_vector,vector_result)
        minheap.insert(doc_score[min_doc_ID],min_doc_ID)
        min_doc_ID = -1

    minheap.minHeap()
    return minheap


def simple_query(query):
    exported_tokens_temp = Tokenizer.get_normalized_tokens(query)
    results_by_order = [] #array of total token length
    current_tokens_index = []
    founded_inverted_index = []
    exported_tokens = []
    for i in exported_tokens_temp:
        if inverted_index.keys().__contains__(i):
            exported_tokens.append(i)

    for i in range(len(exported_tokens)):
        results_by_order.append([])
        current_tokens_index.append(0)
        if inverted_index.keys().__contains__(exported_tokens[i]):
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
        min_doc_ID = -1
        selected_doc_frequency = 0

    return results_by_order.__reversed__()





def main():
    total_words_count = 0
    total_words_count +=1
    # test = "hi?he-----------llo, +-123****,bye///\\\\\\\didi^hmm"
    #
    # out = Tokenizer.get_tokens(test)
    #
    # for i in out:
    #     print(i)

    # test = "بدترین"
    # print(Tokenizer.token_normalizer(test))
    verb_heritages_path = "verbs.txt"
    Tokenizer.make_verb_to_heritage_dict(verb_heritages_path)
    # input()

    id_col = "id"
    content_col = "content"
    url_col = "url"
    data = add_new_file("IR_Spring2021_ph12_7k.xlsx","id","content","url")

    clear_most_repeated_tokens()

    for i in tokens:
        print(i)

    while(True):
        query = input("query: ")
        method = int(input("enter method:\n 0: simple query\n 1: tfidf query 3: champion list"))
        print("Results:")
        if method == 0:
            index = 0
            for answer in query_handler(query,method):
                text = ""
                for i in answer:
                    print("ID: " + str(i) + " --> " + str(urls[i]))
                    index += 1
                    if index > k:
                        break
                if index > k:
                    break
        else:
            output = query_handler(query,method)
            inidex = 0
            for index in range(k):
                currentID = output.remove()
                print("ID: "+ str(currentID) + " --> " + str(urls[currentID]))




if __name__ == "__main__":
    main()