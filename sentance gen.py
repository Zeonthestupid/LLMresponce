import random

def create_random_sentence(file_path, min_words=5, max_words=15):
    # Read all lines from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]

    # Generate a random sentence length
    sentence_length = random.randint(min_words, max_words)

    # Randomly select lines to form the sentence
    random_sentence = ' '.join(random.choice(lines) for _ in range(sentence_length))

    return random_sentence

def generate_sentences(input_file, output_file, num_sentences=1000):
    # Generate sentences and append them to the output file
    with open(output_file, 'a', encoding='utf-8') as file:
        for _ in range(num_sentences):
            sentence = create_random_sentence(input_file)
            file.write(sentence + '\n')

# Example usage
input_file = 'old/wordlist.txt'  # Replace 'your_file.txt' with the path to your text file
output_file = 'test.txt'  # The file where the generated sentences will be saved
generate_sentences(input_file, output_file)