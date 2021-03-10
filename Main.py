import PreProcessing as prep
import LanguageModeling as lmod

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


# make a dictionary of words from the data set    

def create_dictionary(padded_data):
    word_dict = dict()
        
    for word in padded_data:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] =  word_dict[word] + 1                
        
    return word_dict


# make a bigram dictionary by combining two consecutive words from the data 

def collect_bigrams(data):
    
    bigram_dict = dict()
    
    for element in range(0, len(data) - 1):
        bigram = data[element] + " " + data[element + 1]
        
        if bigram not in bigram_dict:
            bigram_dict[bigram] = 1
        else:
            bigram_dict[bigram] += 1
    return bigram_dict
	
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
    train_dict = create_dictionary(padded_train_data)
    processed_train_data = mark_unknown_words(1, padded_train_data, train_dict)
    processed_train_dict = create_dictionary(processed_train_data)
    
    padded_test_data = pad_file_data(test_file)
    test_dict = create_dictionary(padded_test_data)
    processed_test_data = mark_unknown_words(2, padded_test_data, processed_train_dict)
    processed_test_dict = create_dictionary(processed_test_data)
        
    return padded_train_data, train_dict, processed_train_data, processed_train_dict, padded_test_data, test_dict, processed_test_data, processed_test_dict

	

def results_prior_training(dictionary):
    
    print('The number of Word Types in the training corpus:', len(dictionary))
    print('The number of Word Tokens in the training corpus:', sum(dictionary.values()))
    
    return len(dictionary), sum(dictionary.values())


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
'''
def train_unigram_model(test_sentence, train_dict, test_dict):
   
    print('\n\t\t\tGeneral Info prior training Unigram Model:\n')       
   # compute_unigram_log_probability(1, processed_train_dict, test_sentence)    
    return

def compute_probability(model_id, pass_id, given_data, train_data, train_dict):
	probability = 1.0
	
	if model_id == 'unigram':
		token_count = sum(train_dict.values())
		parameters = []
		fractions = []
		
		if pass_id == 1:
        
			for word in data:
				parameter = train_dict[word]
				fraction = parameter / token_count
				parameters.append(parameter)
				fractions.append(fraction)
				probability *= ( parameter / token_count)
				
		elif pass_id == 2:
        
			for word in data.split():
				probability *= (train_dict[word] / token_count)
		
		if pass_id == 1:
        
			print("Parameters needed for computation of unigram probability:")
			print("Splitted sentence and corresponding values from the dictionary:")
			print(data)
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
			for word in range(0, len(data) - 1):
				data_bigram = data[word] + " " + data[word + 1]
				data_bigrams.append(data_bigram)
				
				if data_bigram not in train_bigrams:
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
						parameter_top = train_bigrams[data_bigram]
					
						top_parameters.append(parameter_top)
						
						parameter_bottom = train_dict[data[word]]
						bottom_parameters.append(parameter_bottom)
						
						fraction = (parameter_top / parameter_bottom)
						fractions.append(fraction)
						
						probability *= fraction
							
					elif model_id == "smoothed":
						parameter_top = (train_bigrams[data_bigram] + 1.0)
						top_parameters.append(parameter_top)
				
					else:
						return
						
				if model_id == 'smoothed':
					parameter_bottom = (train_dict[data[word]] + len(train_dict))
					bottom_parameters.append(parameter_bottom)
					
					fraction = (parameter_top / parameter_bottom)
					fractions.append(fraction)
					
					probability *= fraction
				
	
		elif pass_id == 2:
		
			splitted_line = data.split()
        
			for word in range(0, len(splitted_line) - 1):
				data_bigram = (splitted_line[word] + " " + splitted_line[word + 1])
				if data_bigram not in train_bigrams:
					if model_id == 'bigram':
						return 0
					elif model_id == 'smoothed':
						probability *= (1.0 / (train_dict[splitted_line[word]] + len(train_dict)))
				else:
					if model_id == 'bigram':
						probability *= (train_bigrams[data_bigram] / train_dict[splitted_line[word]])
					elif model_id == 'smoothed':
						probability *= ((train_bigrams[data_bigram] + 1.0) / (train_dict[splitted_line[word]] + len(train_dict)))
	
	return probability


def compute_log_probability(model_id, pass_id, given_data, train_data, train_dict):

	probability = 0.0
	processed_data = given_data
	
	if pass_id == 1:
		processed_data = process_given_sentence(given_data)
		
	if model_id == 'unigram':
		model_probability = compute_unigram_probability(model_id, pass_id, processed_data, train_dict)
			
	elif model_id == "bigram":
		model_probability = compute_bigram_probability(model_id, pass_id, processed_data, train_dict)
		
	elif model_id == "smoothed":
		model_probability = compute_smoothed_bigram_probability(model_id, pass_id, processed_data, train_bigrams, train_dict)
		
	else:
		return "undefined"
		
	if model_probability == 0.0
		return "undefined"
		
	else:
		probability += math.log(model_probability, 2)
		
		
	return probability
'''
    
    
    
    

def main():
    
    padded_train_data = pre_process_data()[0]
    train_dict = pre_process_data()[1]
    processed_train_data = pre_process_data()[2]
    processed_train_dict = pre_process_data()[3]
    padded_test_data = pre_process_data()[4]
    test_dict = pre_process_data()[5]
    processed_test_data = pre_process_data()[6]
    processed_test_dict = pre_process_data()[7]
    
    

if __name__== '__main__':
    main()
