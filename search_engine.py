from tokenizer import Tokenizer
import pandas as pd
import copy
import math
from MinHeap import MinHeap
import numpy as np
import sys
import random
import time
import pickle
import gc
min_high_frequency_tokens = 0.80

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

doc_IDS = []
doc_labels = {} #doc ID maps to label
categories = {} #label to doc IDS
docs_vector_presentation = {}
k_means_centers = []
k_means_clusters = {}
k_means_k_value = 5 #K cluster

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


# KNN algorithm for guessing new data label


def KNN_validation(folds = 10,k = 3,fraction_of_labeled_data = 0.03):
    print("KNN validation test started")
    selected_labeled_data = random.sample(list(doc_labels.keys()),int(fraction_of_labeled_data*len(doc_labels)))
    trainning_data = []
    labeled_len = len(selected_labeled_data)
    steps = int(labeled_len / 10)
    accuracy = 0
    k_nearest = []# [(index,similarity)]
    min_similarity = sys.maxsize
    min_index = 0

    similarity = 0
    correct_guessed_lables = 0

    founded = 0
    for f in range(folds):
        checked_docs = 0
        correct_guessed_lables = 0
        trainning_data.clear()

        trainning_data = selected_labeled_data[steps * (f):steps * (f+1)]
        for id in selected_labeled_data:
            if id in trainning_data:
                continue
            my_label = doc_labels[id]
            v1 = get_vector_presentation(tokens,id)
            for train_doc in trainning_data:
            
                v2 = get_vector_presentation(tokens,train_doc)
                similarity = cosin_sim_calculator(v1,v2)

                if founded < k:
                    k_nearest.append((train_doc,similarity))
                    if min_similarity > similarity:
                        min_similarity = similarity
                        min_index = founded
                    founded += 1
                elif min_similarity < similarity:
                    min_similarity = similarity
                    k_nearest.pop(min_index)
                    k_nearest.append((train_doc,similarity))
                    for index in range(len(k_nearest)):
                        i,value = k_nearest[index]
                        if value < min_similarity:
                            min_index = index
                            min_similarity = value
            k_nearest_labels = {}
            best_choice_label_name = None #best label with highest repeat name
            best_choice_number = -1 #best label number of repeated

            for knn_id, value in k_nearest:
                label = doc_labels[knn_id]
                # print(label)
                if not k_nearest_labels.keys().__contains__(label):
                    k_nearest_labels[label] = 0
                k_nearest_labels[label] += 1
                if k_nearest_labels[label] > best_choice_number:
                    best_choice_label_name = label
                    best_choice_number = k_nearest_labels[label]


            guessed_label = best_choice_label_name
            # print(guessed_label)
            random_labels = []
            for i in k_nearest_labels.keys():
                if k_nearest_labels[i] == best_choice_number:
                    random_labels.append(i)
            if len(random_labels) > 1:
                random_select = random_labels[np.random.randint(0,len(random_labels))]
                guessed_label = random_select
                print(guessed_label)

            checked_docs += 1
            # print(guessed_label)
            # print(my_label + "--------" + guessed_label)
            if my_label == guessed_label:
                correct_guessed_lables += 1
        accuracy += correct_guessed_lables/checked_docs
    accuracy = accuracy / folds
    print(str(folds) + "cross fold validation accuracy = " + str(accuracy) + " for k = " + str(k))
    return accuracy



# find all docs without label and choose the best label for them
def KNN(k,n=220): #check with k closest n = number od train data to check new data with


    k_nearest = []# [(index,similarity)]
    min_similarity = sys.maxsize
    min_index = 0

    best_row = 0
    similarity = 0
    correct_guessed_lables = 0

    founded = 0
    sample_train_data = []
    n = int(n/(len(categories)))
    for i in categories:
        sample_train_data.extend(random.sample(categories[i],min(n,len(categories[i]))))
    progress = 0
    docs_len = len(doc_IDS)
    minheap = MinHeap(len(sample_train_data))
    for id in doc_IDS:
        if doc_labels.keys().__contains__(id):
            continue

        # print("new")
        k_nearest.clear()
        founded = 0
        v1 = docs_vector_presentation[id]

        del minheap
        gc.collect()
        minheap = MinHeap(len(sample_train_data))
        for train_doc in sample_train_data:

            v2 = docs_vector_presentation[train_doc]
            similarity = cosin_sim_calculator(v1,v2)
            minheap.insert(-similarity,train_doc)
            # if founded < k:
            #     k_nearest.append((train_doc,similarity))
            #     if min_similarity > similarity:
            #         min_similarity = similarity
            #         min_index = founded
            #     founded += 1
            # elif min_similarity < similarity:
            #     min_similarity = similarity
            #     k_nearest.pop(min_index)
            #     k_nearest.append((train_doc,similarity))
            #     for index in range(len(k_nearest)):
            #         i,value = k_nearest[index]
            #         if value < min_similarity:
            #             min_index = index
            #             min_similarity = value
        k_nearest_labels = {}
        best_choice_label_name = None #best label with highest repeat name
        best_choice_number = -1 #best label number of repeated

        minheap.minHeap()

        for i in range(k):
            k_nearest.append(minheap.remove())
        for knn_id in k_nearest:
            label = doc_labels[knn_id]
            # print(label)
            if not k_nearest_labels.keys().__contains__(label):
                k_nearest_labels[label] = 0
            k_nearest_labels[label] += 1
            if k_nearest_labels[label] > best_choice_number:
                best_choice_label_name = label
                best_choice_number = k_nearest_labels[label]


        guessed_label = best_choice_label_name
        random_labels = []
        for i in k_nearest_labels.keys():
            if k_nearest_labels[i] == best_choice_number:
                random_labels.append(i)
        if len(random_labels) > 1:
            random_select = random_labels[np.random.randint(0,len(random_labels))]
            guessed_label = random_select

        # print(my_label + "-----" + guessed_label)

        doc_labels[id] = guessed_label
        if not categories.keys().__contains__(guessed_label):
            categories[guessed_label] = []
        categories[guessed_label].append(id)
        progress += 1
        print("progress : " + str(progress / docs_len))
        

# KNN--------------------------------------------


# Kmeans
# calculates the k best centers and fill the clusters according to that
def k_means(k, max_itr,doc_IDs = doc_IDS):

    print("K means started")
    random_centers = list()

    clusters = list()
    similarity_value_clusters = []

    for i in range(k):
        print(len(doc_IDs))
        random_centers.append(doc_IDs[random.randint(0, len(doc_IDs))])
        clusters.append(list())
        similarity_value_clusters.append([])


    itr = 0
    centers = random_centers
    centers_vector_presentation = []
    while itr < max_itr:
        row_index = 0
        for i in clusters:
            i.clear()
        for i in centers:
            centers_vector_presentation.append(docs_vector_presentation[i])
        new_centers = copy.deepcopy(centers)
        for row in doc_IDs:

            # row = row_data.values.tolist()
            best_match_index = 0
            best_similarity = -1
            for c in range(k):
                sim = cosin_sim_calculator(centers_vector_presentation[c],docs_vector_presentation[row])

                if best_similarity < sim:
                    best_similarity = sim
                    best_match_index = c
            clusters[best_match_index].append(row)
            similarity_value_clusters[best_match_index].append(best_similarity)
            row_index += 1

        last_index =0
        #setting new centers values
        for i in range(k):
            if len(clusters[i]) == 0:
                continue
            cluster_points = clusters[i]
            mean_similarity_value = np.mean(np.array(similarity_value_clusters[i]))
            best_index = 0
            closest_to_mean = abs(mean_similarity_value - similarity_value_clusters[i][0])
            index = 0
            for j in similarity_value_clusters[i]:
                new_sim = abs(j - mean_similarity_value)
                if new_sim < closest_to_mean:
                    best_index = index
                    closest_to_mean = new_sim
                index += 1

            new_centers[i] = cluster_points[best_index]
        # print("new centers : \n",new_centers)
        # print("old centers: \n",centers)

        if centers == new_centers:
            break
        centers = new_centers
        itr+=1
    for i in clusters:
        print(i)

    k_means_centers = centers
    indexx = 0
    for i in k_means_centers:
        k_means_clusters[i] = clusters[indexx]
        indexx += 1

#Kmeans-----------------------------------------------------------

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



def add_new_file(path,content_ID_col_name,content_col_name,url_col_name,label_col = None):
    data = pd.read_excel(path)
    total = copy.deepcopy(total_docs)
    for index, row in data.iterrows():
        ID = row[content_ID_col_name] + total
        line = row[content_col_name]
        urls[ID] = row[url_col_name]
        doc_IDS.append(ID)
        # statement = "S " + i
        # print(ID)
        if label_col != None:
            label = row[label_col]
            doc_labels[ID] = label
            if not categories.keys().__contains__(label):
                categories[label] = []
            categories[label].append(ID)


        current_tokens = Tokenizer.get_tokens(line)
        for token in current_tokens:
            if token in ignorance_tokens:
                continue
            normalized_token = Tokenizer.token_normalizer(token)
            update_inverted_index_list(normalized_token,ID,True)
            update_term_frequency(normalized_token,token,line,ID)


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
            df.pop(token)
            for i in tdf:
                if tdf[i].keys().__contains__(token):
                    tdf[i].pop(token)
    copy_tokens.clear()


def champion_tfidf_query(query,use_heap = True,number_outputs = k):
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

    doc_results = []
    founded = 0
    min_index = 0
    min_value = 2

    doc_score = {}
    if use_heap:
        minheap = MinHeap(r)

    for i in all_docs:
        vector_result = get_vector_presentation(exported_tokens, i)
        doc_score[i] = cosin_sim_calculator(query_vector,vector_result)
        if use_heap:
            minheap.insert(-doc_score[i],i)
        else:
            if founded < number_outputs:
                doc_results.append(i)
                if min_value > doc_score[i]:
                    min_value = doc_score[i]
                    min_index = founded
                founded += 1
            elif min_value < doc_score[i]:
                min_value = doc_score[i]
                doc_results.pop(min_index)
                doc_results.append(i)
                for j in range(len(doc_results)):
                    if doc_score[doc_results[j]] < min_value:
                        min_index = j
                        min_value = doc_score[doc_results[j]]

    if use_heap:
        minheap.minHeap()
        if len(all_docs) > number_outputs:
            for i in range(number_outputs):
                doc_results.append(minheap.remove())
        else:
            for i in range(len(all_docs)):
                doc_results.append(minheap.remove())

    if len(all_docs) < number_outputs:
        new_data,new_doc_score = tfIdf_cosine_query(query,use_heap,number_outputs=number_outputs - len(all_docs),exceptions= doc_results)
        for i in new_data:
            doc_results.append(i)
        for i in new_doc_score.keys():
            doc_score[i] = new_doc_score[i]

    return doc_results,doc_score





def get_vector_presentation(terms,doc_id):
    vector_result = []
    for term in terms:
        vector_result.append(tfIdf_calculator(term, doc_id))
    return vector_result

# number_cluster_check= centers of clusters to check their followers
def query_handler(query,method = 0,use_heap=False,number_outputs = k,category = None,number_cluster_check=3):
    start = time.process_time()
    if method == 0:
        return simple_query(query)
    if method == 1:
        return tfIdf_cosine_query(query,use_heap,number_outputs,only_selected_docs=only_selected_docs)
    if method == 2:
        return champion_tfidf_query(query,use_heap,number_outputs)
    if method == 3:
        return cluster_search(query,number_cluster_check,use_heap=use_heap,number_outputs=number_outputs)
    if method == 4:
        return in_category_search(query,category,use_heap=use_heap,number_outputs=number_outputs)
    elapsed_time = time.process_time() - start

    print("time to answer query: " + str(elapsed_time))


def limited_doc_search(query, selected_docs, use_heap = False, number_outputs=k):
    exported_tokens_temp = Tokenizer.get_normalized_tokens(query)
    exported_tokens = []

    query_vector = []

    for i in exported_tokens_temp:
        if champion_list.keys().__contains__(i):
            print(i)
            exported_tokens.append(i)
            query_vector.append(tfIdf_calc(i, 1))
    doc_results = []
    founded = 0
    min_index = 0
    min_value = 2

    doc_score = {}
    if use_heap:
        minheap = MinHeap(r)

    for i in selected_docs:
        vector_result = get_vector_presentation(exported_tokens, i)
        doc_score[i] = cosin_sim_calculator(query_vector, vector_result)
        if use_heap:
            minheap.insert(-doc_score[i], i)
        else:
            if founded < number_outputs:
                doc_results.append(i)
                if min_value > doc_score[i]:
                    min_value = doc_score[i]
                    min_index = founded
                founded += 1
            elif min_value < doc_score[i]:
                min_value = doc_score[i]
                doc_results.pop(min_index)
                doc_results.append(i)
                for j in range(len(doc_results)):
                    if doc_score[doc_results[j]] < min_value:
                        min_index = j
                        min_value = doc_score[doc_results[j]]

    if use_heap:
        minheap.minHeap()
        if len(selected_docs) > number_outputs:
            for i in range(number_outputs):
                doc_results.append(minheap.remove())
        else:
            for i in range(len(selected_docs)):
                doc_results.append(minheap.remove())

    return doc_results, doc_score

def cluster_search(query,number_centers,use_heap = False,number_outputs=k):
    selected_centers = limited_doc_search(query,k_means_centers,use_heap=use_heap,number_outputs=number_centers)
    founded_docs = []
    for i in selected_centers:
        founded_docs.extend(k_means_clusters[i])
    return limited_doc_search(query,founded_docs,use_heap=use_heap,number_outputs=number_outputs)

def in_category_search(query,category,use_heap = False,number_outputs=k):
    only_selected_docs = categories[category]
    return limited_doc_search(query,only_selected_docs,use_heap=use_heap,number_outputs=number_outputs)


# calculates the vector space presentation and similarity and sort by heap(return the sorted MinHeap)
def tfIdf_cosine_query(query,use_heap = False,number_outputs = k,exceptions = []):
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



    #results array
    doc_results = []
    founded = 0
    min_index = 0
    min_value = 2
    while (True):
        for index in range(len(exported_tokens)):
            token_doc_index = current_tokens_index[index]
            if len(founded_inverted_index[index]) > token_doc_index:
                if min_doc_ID == -1 or min_doc_ID > founded_inverted_index[index][token_doc_index]:
                    min_doc_ID = founded_inverted_index[index][token_doc_index]

        if min_doc_ID == -1:
            break


        # calculates the vector space presentation of doc tfIdf for selected ID

        for index in range(len(exported_tokens)):
            token_doc_index = current_tokens_index[index]
            if len(founded_inverted_index[index]) > token_doc_index:
                if min_doc_ID == founded_inverted_index[index][token_doc_index]:
                    current_tokens_index[index] += 1

        vector_result = get_vector_presentation(exported_tokens,min_doc_ID)
        doc_score[min_doc_ID] = cosin_sim_calculator(query_vector,vector_result)
        if use_heap:
            minheap.insert(-doc_score[min_doc_ID],min_doc_ID)
        elif min_doc_ID not in exceptions:
            if founded < number_outputs:
                doc_results.append(min_doc_ID)
                if min_value > doc_score[min_doc_ID]:
                    min_value = doc_score[min_doc_ID]
                    min_index = founded
                founded += 1
            elif min_value < doc_score[min_doc_ID]:
                min_value = doc_score[min_doc_ID]
                doc_results.pop(min_index)
                doc_results.append(min_doc_ID)
                for j in range(len(doc_results)):
                    if doc_score[doc_results[j]] < min_value:
                        min_index = j
                        min_value = doc_score[doc_results[j]]


        min_doc_ID = -1

    if use_heap:
        minheap.minHeap()
        doc_results.clear()
        for i in range(number_outputs):
            doc_results.append(minheap.remove())

    return doc_results,doc_score



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


def calculate_docs_vector_presentation(all_doc_IDs, terms):
    docs_vector_presentation.clear()
    for i in all_doc_IDs:
        docs_vector_presentation[i] = get_vector_presentation(terms, i)




def save_data():
    pickle_out = open("categories.pickle", "wb")
    pickle.dump(categories, pickle_out)
    pickle_out.close()

def read_data():
    pickle_in = open("dict.pickle", "rb")
    categories = pickle.load(pickle_in)

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

    id_col = "id"
    content_col = "content"
    url_col = "url"
    label_col = "topic"
    add_new_file("IR00_3_11k News.xlsx",id_col,content_col,url_col,label_col)



    clear_most_repeated_tokens()
    for i in tokens:
        print(i + " --- " + str((len(i) / len(tokens))))
    print(len(tokens))
    # make_championList()
    calculate_docs_vector_presentation(doc_IDS, tokens)

    # KNN_validation(10)
    KNN(7)
    save_data()
    return None

    # k_means(k_means_k_value,6)

    
    while(True):
        try:
            query = input("query: \n")
            if query.startswith("cat:"):
                query = query.replace("cat:","")
                cat = query.split(" ")[0]
                if not categories.keys().__contains__(cat):
                    print("wrong category!")
                    continue
                use_heap = int(input("\n enter 1 in order to use min heap or enter 0 otherwise")) is 1
                print(use_heap)
                output,values = query_handler(query,4,use_heap=use_heap,category=cat)
                for i in output:
                    print("ID: "+ str(i)+", score: " + str(values[i]) + " --> " + str(urls[i]))


            method = int(input("enter method:\n 0: simple query\n 1: tfidf query\n 2: champion list\n 3: K_means check\n"))
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
            elif method == 1 or method == 2:
                use_heap = int(input("\n enter 1 in order to use min heap or enter 0 otherwise")) is 1
                print(use_heap)
                output,values = query_handler(query,method,use_heap=use_heap)
                for i in output:
                    print("ID: "+ str(i)+", score: " + str(values[i]) + " --> " + str(urls[i]))
            elif method == 3:
                use_heap = int(input("\n enter 1 in order to use min heap or enter 0 otherwise")) is 1
                number_of_cluster_centers = int(input("\n enter number od clusters to check"))

                output,values = query_handler(query,method,use_heap=use_heap,number_outputs=number_of_cluster_centers)
                for i in output:
                    print("ID: "+ str(i)+", score: " + str(values[i]) + " --> " + str(urls[i]))
        except:
            print("some ERRORS accrued")




if __name__ == "__main__":
    main()