
from tokenizer import Tokenizer









def main():
    test = "hi?hello-bye/didi^hmm"

    out = Tokenizer.get_tokens(test)

    for i in out:
        print(i)





if __name__ == "__main__":
    main()