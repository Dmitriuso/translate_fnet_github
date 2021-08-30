import spacy
from spacy.lang.de.examples import sentences

nlp = spacy.load("de_dep_news_trf")
doc = nlp(sentences[0])

def split_trg(sentence):
    splitted_list = []
    j = 0
    for i, s in enumerate(sentence):
        if s in ['.', ':', ';', '-', '?', '!', ",", "'"]: 
            splitted_list.append(sentence[j:i])
            splitted_list.append(s)
            j = i+1
        elif s == " ":
            splitted_list.append(sentence[j:i])
            j = i+1
    splitted_list = [s for s in splitted_list if s not in ['', ' ', None]]
    return splitted_list


if __name__ == '__main__':
    print(doc.text)
    for token in doc:
        print(token.text, token.pos_, token.dep_)
