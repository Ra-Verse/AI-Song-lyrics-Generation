import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import os
import re

# Directory containing artist lyrics files
folder_path = "Artists"

# List of available artists (sorted alphabetically)
artists = [
    "Adele", "Al Green", "Alicia Keys", "Amy Winehouse", "Best Songs 1900s", "Björk", "Blink-182",
    "Bob Dylan", "Bob Marley", "Britney Spears", "Bruce Springsteen", "Bruno Mars", "Cake", "Disney",
    "DJ Khaled", "Dolly Parton", "Dr. Seuss", "Drake", "Eminem", "Eminem (experimental)", "Emily Dickinson", "Hamilton",
    "Janis Joplin", "Jimi Hendrix", "Johnny Cash", "Joni Mitchell", "Justin Bieber", "Kanye", "Kanye West",
    "Lady Gaga", "Leonard Cohen", "Lil Wayne", "Lin-Manuel Miranda", "Lorde", "Ludacris", "Michael Jackson",
    "Missy Elliott", "Nickelback", "Nicki Minaj", "Nirvana", "Notorious BIG", "Nursery Rhymes", "Patti Smith",
    "Paul Simon", "Prince", "Radiohead", "R. Kelly", "Rihanna", "Taylor Swift", "The Beatles", "The Notorious B.I.G."
]

print("Available Artists:")
for i, artist in enumerate(artists, 1):
    print(f"{i}. {artist}")
print("52. All Artists (This will need much more memory and time)")
print("\n'All-songs.txt' contains data from all these artists.")

# Get user selection
while True:
    choice = input("Enter the number corresponding to an artist: ")
    if choice.isdigit() and 1 <= int(choice) <= len(artists):
        choice = int(choice) - 1
        filename = f"{artists[choice].replace(' ', '-').lower()}.txt"
        break
    elif choice == "52":
        filename = "All-songs.txt"
        break
    else:
        print("Invalid choice. Please enter a valid number.")

filepath = os.path.join(folder_path, filename)



# Read and validate file length
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = re.sub(r' +', ' ', text)  # Replace multiple spaces with a single space
text = text.strip()  # Remove leading/trailing whitespace

if len(text) < 150000:
    proceed = input("The file has less than 150,000 characters. Do you want to continue? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Exiting program.")
        exit()


text = text[0:5000000] # Adjust according to the memory available

characters = list(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 120
STEP_SIZE = 3

# sentences = []
# next_characters = []
#
# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i + SEQ_LENGTH])
#     next_characters.append(text[i + SEQ_LENGTH])
#
# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.uint8)
# y = np.zeros((len(sentences), len(characters)), dtype=np.uint8)
#
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         x[i, t, char_to_index[char]] = 1
#     y[i, char_to_index[next_characters[i]]] = 1
#
# model = Sequential()
# model.add(LSTM(256, input_shape=(SEQ_LENGTH, len(characters)), return_sequences=True))  # First LSTM layer
# model.add(LSTM(128))  # Second LSTM layer
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0005))
# model.fit(x, y, batch_size=256, epochs=3, verbose=4)
#
# model.save('model.h5')

model = tf.keras.models.load_model('model.h5')

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated_text = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated_text += sentence

    for _ in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated_text += next_char
        sentence = sentence[1:] + next_char

    return generated_text

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

while True:
    print(
        "\nTemperature controls the randomness of the text generation. Lower values (e.g., 0.25) make the text more predictable, while higher values (e.g., 1.0) make it more random.")
    temp = float(input("Enter the temperature (0.25 - 1.0): "))
    length = int(input("Enter the number of characters to generate: "))
    print(generate_text(length, temperature=temp))
    cont = input("Generate again? (yes/no): ").strip().lower()
    if cont != 'yes':
        break