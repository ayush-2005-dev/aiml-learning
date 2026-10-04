from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
sentence = "私は学校に行きます"

for token in tokenizer.tokenize(sentence):
    print(token.surface, "|", token.part_of_speech.split(",")[0:2])