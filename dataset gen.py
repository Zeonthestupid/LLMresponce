with open('text.txt', 'r', encoding='utf-8') as input_file, \
     open('user_inputs.txt', 'w', encoding='utf-8') as user_inputs_file, \
     open('responses.txt', 'w', encoding='utf-8') as responses_file:
    # Iterate over each line in the input file
    for i, line in enumerate(input_file):
        # Strip any leading/trailing whitespace from the line
        line = line.strip()
        # Write the line to the appropriate output file
        if i % 2 == 0:
            user_inputs_file.write(line + '\n')
        else:
            responses_file.write(line + '\n')
