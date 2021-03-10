import math

filepath = input("Provide file location with training and testing files: ")
train_file = filepath + "/train.txt"
test_file = filepath + "/test.txt"

global padded_train_data, train_dict, processed_train_data, processed_train_dict
global padded_test_data, test_dict, processed_test_data, processed_test_dict

padded_train_data = []
train_dict = dict()
processed_train_data = []
processed_train_dict = dict()
        
padded_test_data = []
test_dict = dict()
processed_test_data = []
processed_test_dict = dict()


############################### Pre-Proocessing ###############################

# make all letters of the entire document lowercase
# pad each distinct sentence in the data set with start (<s>) 
#       and end (</s>) symbols
def pad_file_data(input_file):
    
    padded_data = [] 
    file = open(input_file, 'r', encoding = 'utf-8')
        
    for line in file:
        processed_line = line.strip().lower().split()
            
        processed_line.insert(0, '<s>')
        processed_line.insert(len(processed_line), '</s>')
    
        for single_element in processed_line:
            padded_data.append(single_element)
    
    file.close()
    return padded_data 

# make dictionary from given data

def make_dictionary(pass_id, given_data):
    dictionary = dict()
    
   
    if pass_id == 'regular':
        # make a dictionary of words from the data set
        for word in given_data:
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1
        
    elif  pass_id == 'bigram':
        # make a bigram dictionary by combining two consecutive words from the data
        for element in range(0, len(given_data) - 1):
            bigram = given_data[element] + ' ' + given_data[element + 1]
        
            if bigram not in dictionary:
                dictionary[bigram] = 1
            else:
                dictionary[bigram] += 1
    
    return dictionary
    
# mark words with unknown symbol (<unk>) if they are seen no more than once
def mark_unknown_words(pass_id, data, file_dict):
        
    marked_data = []
        
    if pass_id == 1:
        for word in data:
            if file_dict[word] == 1:
                marked_data.append('<unk>')
            else:
                marked_data.append(word)
                    
    elif pass_id == 2:
        for word in data:
            if word not in file_dict:
                marked_data.append('<unk>')
            else:
                marked_data.append(word)
                    
    else:
        return
        
    return marked_data


def pre_process_data(train_file, test_file):
    
    padded_train_data = pad_file_data(train_file)
    train_dict = make_dictionary('regular', padded_train_data)
    processed_train_data = mark_unknown_words(1, padded_train_data, train_dict)
    processed_train_dict = make_dictionary('regular', processed_train_data)
    
    padded_test_data = pad_file_data(test_file)
    test_dict = make_dictionary('regular', padded_test_data)
    processed_test_data = mark_unknown_words(2, padded_test_data, processed_train_dict)
    processed_test_dict = make_dictionary('regular', processed_test_data)
        
    return padded_train_data, train_dict, processed_train_data, processed_train_dict, padded_test_data, test_dict, processed_test_data, processed_test_dict

    
# output results: number of types and total number of tokens
def results_prior_training(dictionary):
    
    print('The number of Word Types in the training corpus:', len(dictionary))
    print('The number of Word Tokens in the training corpus:', sum(dictionary.values()))
    
    return len(dictionary), sum(dictionary.values())


# Count the number of unoccurred tokens by checking the training dictionary
#   Get the percentage of unoccurred tokens by dividing the unoccured token
#   count by the total number of tokens in the test dictionary
# Get the percentage of unoccured tokens by multiplying the fraction obtained
# above by 100
#
# Count the number of unoccured types by dividing the length of the dictionary 
# with unoccured tokens by the length of the test dictionary
# Get the percentage of unoccurred types by multiplying the fraction obtained
# above by 100
def get_percentage_unoccurred_tokens_and_types(train_dict, test_dict):
    
    percentage_unoccurred_tokens = 0
    percentage_unoccurred_types = 0
        
    unoccurred_test_dict = dict()
    
    for word in test_dict:
        if word not in train_dict:
            unoccurred_test_dict[word] = test_dict[word]
            
            
    percentage_unoccurred_tokens = ((sum(unoccurred_test_dict.values())/sum(test_dict.values())) * 100)
    percentage_unoccurred_types = ((len(unoccurred_test_dict)/len(test_dict)) * 100)
    
    return percentage_unoccurred_tokens, percentage_unoccurred_types

# Count the number of unoccurred bigram tokens by checking the training 
# bigram dictionary
#   Get the percentage of unoccurred bigram tokens by dividing the unoccured 
#   bigram token count by the total number of tokens in the test dictionary
# Get the percentage of unoccured bigram tokens by multiplying the fraction
# obtained above by 100
#
# Count the number of unoccured bigram types by dividing the length of the  
# dictionary with unoccured bigram tokens by the length of the test bigram 
# dictionary
# Get the percentage of unoccurred bigram types by multiplying the fraction obtained
# above by 100
def get_percentage_unoccurred_bigrams_tokens_and_types(train_bigram_dict, test_bigram_dict, test_dict):
    
    percentage_unoccurred_bigram_tokens = 0
    percentage_unoccurred_bigram_types = 0
    
    unoccurred_bigram_test_dict = dict()
    
    for bigram in test_bigram_dict:
        if bigram not in train_bigram_dict:
            if bigram not in unoccurred_bigram_test_dict:
                unoccurred_bigram_test_dict[bigram] = 1
            else: 
                unoccurred_bigram_test_dict[bigram] += 1
    
    percentage_unoccurred_bigram_tokens = ( (sum( unoccurred_bigram_test_dict.values() ) / sum(test_bigram_dict.values() ) ) * 100 )
    percentage_unoccurred_bigram_types = ( (len( unoccurred_bigram_test_dict )/ len( test_bigram_dict ) ) * 100 )
    
    return percentage_unoccurred_bigram_tokens, percentage_unoccurred_bigram_types


# for a given sentence split the words, lowercase each of them,
# and add start and end symbol
def process_given_sentence(given_data):
    
    data = given_data.strip()
    data = data.lower()
    splitted_data = data.split()
    splitted_data.insert(0, '<s>')
    splitted_data.insert(len(splitted_data), '</s>')
    
    return splitted_data


'''
The following functions use variable pass_id.
pass_id  = 1 - corresponds to analysiis of models based on a given sentence
pass_id = 2 - corresponds to abalysis of models based on the entire test data
'''
def compute_probability(model_id, pass_id, given_data, train_data, train_dict):
    probability = 1.0
    
    if model_id == 'unigram':
        token_count = sum(train_dict.values())
        parameters = []
        fractions = []
        
        if pass_id == 1:
        
            for word in given_data:
                parameter = train_dict[word]
                fraction = parameter / token_count
                parameters.append(parameter)
                fractions.append(fraction)
                probability *= ( parameter / token_count)
                
        elif pass_id == 2:
        
            for word in given_data.split():
                probability *= (train_dict[word] / token_count)
        
        if pass_id == 1:
        
            print("Parameters needed for computation of unigram probability:")
            print("Splitted sentence and corresponding values from the dictionary:")
            print(given_data)
            print(parameters)
            print("Total number of tokens in the train dictionary:", token_count)
            print("Fraction for each parameter that is multiplied by one another later:")
            print(fractions)   
    
    elif model_id == 'bigram' or model_id == 'smoothed':
        data_bigrams = []
        top_parameters = []
        bottom_parameters = []
        fractions = []
    
        if pass_id == 1:
            for word in range(0, len(given_data) - 1):
                data_bigram = given_data[word] + " " + given_data[word + 1]
                data_bigrams.append(data_bigram)
                
                if data_bigram not in train_data:
                    if model_id == 'bigram':
                        print("Parameters needed for computation of bigram probability:")
                        print("Sentence's bigrams and corresponding values from the dictionary:")
                        print(data_bigrams)
                        print(top_parameters)
                        print(bottom_parameters)
                        print("Fraction for each parameter that is multiplied by one another later:")
                        print(fractions)
                        print("Returning Probability without log:", probability)
                        return 0
                    
                    elif model_id == "smoothed":
                        parameter_top = 1.0
                        top_parameters.append(parameter_top)
                
                else:
                    if model_id == 'bigram':
                        parameter_top = train_data[data_bigram]
                    
                        top_parameters.append(parameter_top)
                        
                        parameter_bottom = train_dict[given_data[word]]
                        bottom_parameters.append(parameter_bottom)
                        
                        fraction = (parameter_top / parameter_bottom)
                        fractions.append(fraction)
                        
                        probability *= fraction
                            
                    elif model_id == 'smoothed':
                        parameter_top = (train_data[data_bigram] + 1.0)
                        top_parameters.append(parameter_top)
                
                    else:
                        return
                        
                if model_id == 'smoothed':
                    parameter_bottom = (train_dict[given_data[word]] + len(train_dict))
                    bottom_parameters.append(parameter_bottom)
                    
                    fraction = (parameter_top / parameter_bottom)
                    fractions.append(fraction)
                    
                    probability *= fraction
                
    
        elif pass_id == 2:
        
            splitted_line = given_data.split()
        
            for word in range(0, len(splitted_line) - 1):
                data_bigram = (splitted_line[word] + " " + splitted_line[word + 1])
                if data_bigram not in train_data:
                    if model_id == 'bigram':
                        return 0
                    elif model_id == 'smoothed':
                        probability *= (1.0 / (train_dict[splitted_line[word]] + len(train_dict)))
                else:
                    if model_id == 'bigram':
                        probability *= (train_data[data_bigram] / train_dict[splitted_line[word]])
                    elif model_id == 'smoothed':
                        probability *= ((train_data[data_bigram] + 1.0) / (train_dict[splitted_line[word]] + len(train_dict)))
    
    return probability


def compute_log_probability(model_id, pass_id, given_data, train_data, train_dict):
    probability = 0.0
    model_probability = 0.0
    processed_data = given_data
    if pass_id == 1:
        processed_data = process_given_sentence(given_data)
        
    if model_id == 'unigram':
        
        model_probability = compute_probability(model_id, pass_id, processed_data, train_data, train_dict)
       
        
    elif model_id == 'bigram':
        model_probability = compute_probability(model_id, pass_id, processed_data, train_data, train_dict)
        
    elif model_id == 'smoothed':
        model_probability = compute_probability(model_id, pass_id, processed_data, train_data, train_dict)

    else:
        return 'undefined'
    
    if model_probability == 0.0:
        return 'undefined'
        
    else:
        probability += math.log(model_probability, 2)
    
    return probability

def update_test_data(processed_test_data, output_file):
    output = open(output_file, 'w', encoding = 'utf-8')
    
    data_list = []
    list0 = []
    
    for i in range(0, len(processed_test_data)):
        
        if processed_test_data[i] != '</s>':
            list0.append(processed_test_data[i])
            output.write(processed_test_data[i] + ' ')
        else:
            #list0.append('</s> <//s>')
            list0.append('</s>')
            output.write('</s>')
            #output.write("\n")
            data_list.append(list0)
            list0 = []
    #print(data_list)
    output.close()
    return data_list


def compute_perplexity(pass_id, given_data, model_log_probability):
    
    if model_log_probability == 'undefined':
        return 'undefined'
    else:
        perplexity = 0.0
        
        processed_data = given_data
        if pass_id == 1:
            processed_data = process_given_sentence(given_data)
            
        # m - number of words in the test data
        # l - log probability divided by m (see above)
        
        m = len(processed_data)
        l = model_log_probability / m
        perplexity = 2 ** (-l)
        
        return perplexity
    
    
def compute_data_perpelexity(output_file, language_model, train_dict, test_dict, train_bigrams):
    probability = 1.0
    output = open(output_file, 'r', encoding = 'utf-8')
    data = ''
    
    for line in output:
        line = line.replace('</s>', ' </s> <//s> ')
        data += line
        
    m = 0
    l = 0

    for word in data.split():
        if word != '<//s>':
            m+=1
    for line in data.split('<//s>'):
        if language_model == 'unigram':
            sentence_probability = compute_probability('unigram', 2, line, train_bigrams, train_dict)
            
            if sentence_probability == 0.0:
                return 'undefined'
            else:
                probability += math.log(sentence_probability, 2)
                
        elif language_model == 'bigram':
            sentence_probability = compute_probability('bigram', 2, line, train_bigrams, train_dict)
            if sentence_probability == 0.0:
                return 'undefined'
            else:
                probability += math.log(sentence_probability, 2)
        
        elif language_model == 'smoothed':
            sentence_probability = compute_probability('smoothed', 2, line, train_bigrams, train_dict)
            if sentence_probability == 0.0:
                continue
            else:
                probability += math.log(sentence_probability, 2)
        
    l = probability / m
    perplexity = 2 ** ((-1) * l)
    return perplexity



def main():
    
    pre_processing = pre_process_data(train_file, test_file)
    
    padded_train_data = pre_processing[0]
    train_dict = pre_processing[1]
    processed_train_data = pre_processing[2]
    processed_train_dict = pre_processing[3]
    padded_test_data = pre_processing[4]
    test_dict = pre_processing[5]
    processed_test_data = pre_processing[6]
    processed_test_dict = pre_processing[7]
    
    results = results_prior_training(processed_train_dict)
    unique_train_token_count = results[0]
    total_train_token_count = results[1]
    
    
    percentage_unoccured = get_percentage_unoccurred_tokens_and_types(train_dict, test_dict)
    print('\nPercentage of word tokens in the test corpus that did not occur in the training: {}%'.format('%.2f' %(percentage_unoccured[0])))
    print('Percentage of word types in the test corpus that did not occur in the training: {}%\n'.format('%.2f' %(percentage_unoccured[1])))
    
    
    train_bigrams = make_dictionary('bigram', processed_train_data)
    test_bigrams = make_dictionary('bigram', processed_test_data)
    bigram_percentage_unoccured = get_percentage_unoccurred_bigrams_tokens_and_types(train_bigrams, test_bigrams, processed_test_dict)    
    print('Percentage of bigram tokens in the test corpus that did not occur in the training: {}%'.format('%.2f' %(bigram_percentage_unoccured[0])))
    print('Percentage of bigram types in the test corpus that did not occur in the training: {}%\n'.format('%.2f' %(bigram_percentage_unoccured[1])))
    
    
    given_sentence = 'I look forward to hearing your reply .'
    data = [] 
    unigram_log_probability = compute_log_probability('unigram', 1, given_sentence, data, processed_train_dict)
    
    bigram_log_probability = compute_log_probability('bigram', 1, given_sentence, train_bigrams, processed_train_dict)
    print('Bigram Probability for the given sentence:', bigram_log_probability, '\n')
    
    
    smoothed_bigram_log_probability = compute_log_probability('smoothed', 1, given_sentence, train_bigrams, processed_train_dict)   
    print('Bigram Smoothed Probability for the given sentence:', smoothed_bigram_log_probability, '\n')
    
    
    print("Computation of perplexity for sentence 'I look forward to hearing your reply .'")
    unigram_perplexity = compute_perplexity(1, given_sentence, unigram_log_probability)
    print('Perplexity of the given sentence under Unigram:', unigram_perplexity)

    bigram_perplexity =  compute_perplexity(1, given_sentence, bigram_log_probability)
    print('Perplexity of the given sentence under Bigram:', bigram_perplexity)
    smoothed_bigram_perplexity =  compute_perplexity(1, given_sentence, smoothed_bigram_log_probability)
    print('Perplexity of the given sentence under Bigram with Add-One Smoothing:', smoothed_bigram_perplexity, '\n')
    
    
    updated_data = update_test_data(processed_test_data, 'newText.txt')    
    test_unigram_perplexity = compute_data_perpelexity('newText.txt', 'unigram', processed_train_dict, processed_test_dict, train_bigrams)
    print('Perplexity of the entire test corpus under unigram:', test_unigram_perplexity)
    
    test_bigram_perplexity = compute_data_perpelexity('newText.txt', 'bigram', processed_train_dict, processed_test_dict, train_bigrams)
    print('Perplexity of the entire test corpus under bigram:', test_bigram_perplexity)
    
    test_smoothed_bigram_perplexity = compute_data_perpelexity('newText.txt', 'smoothed', processed_train_dict, processed_train_dict, train_bigrams)
    print('Perplexity of the entire test corpus under bigram with Add-One Smoothing:', test_smoothed_bigram_perplexity)

    
    
if __name__ == "__main__":
    main()
