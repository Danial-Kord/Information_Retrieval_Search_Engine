
from tokenizer import Tokenizer




def add_new_file(path):
    return None





def main():
    test = "hi?he-----------llo, +-123****,bye///\\\\\\\didi^hmm"

    out = Tokenizer.get_tokens(test)

    for i in out:
        print(i)





if __name__ == "__main__":
    main()