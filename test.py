import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import random



with open("encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)

path_to_image_folder = 'Flickr8k/Flickr8k_Dataset/'
model = load_model('model_30.h5')

file = open('vocab.txt', 'r')
text = file.read()
file.close()
vocab = text.split()

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

max_length = 34
def setCaption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

image_test_id = random.randint(0,6000)
image_name = list(encoding_test.keys())[image_test_id]
image_vector = encoding_test[image_name].reshape((1,2048))

x = plt.imread(path_to_image_folder + image_name)
plt.imshow(x)
plt.title(setCaption(image_vector))
plt.show()
