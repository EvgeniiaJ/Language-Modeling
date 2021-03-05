import PreProcessing as prep

filepath = input("Provide file location with trainign and test files: ")
train_file = filepath + "/train.txt"
test_file = filepath + "/test.txt"

class LanguageModeling:
    
    global padded_train_data, train_dict, processed_train_data, processed_train_dict
    global padded_test_data, test_dict, processed_test_data, processed_test_dict
    
    global pre_processing
    
    def pre_process_data():
        
        padded_train_data = pre_processing.pad_file_data(train_file)
        train_dict = pre_processing.create_dict(padded_train_data)
        processed_train_data = pre_processing.mark_unknown_words(1, padded_train_data, train_dict)
        processed_train_dict = pre_processing.create_dict(processed_train_data)
        
        padded_test_data = pre_processing.pad_file_data(test_file)
        test_dict = pre_processing.create_dict(padded_test_data)
        processed_test_data = pre_processing.mark_unknown_words(2, padded_test_data, processed_train_dict)
        processed_test_dict = pre_processing.create_dict(processed_test_data)
        
        return
    
    def main():
        pre_processing = prep() 
        pre_process_data()
        results_prior_training()
        
        single_testing_sentence = 'I look forward to hearing your reply .'
        train_unigram_model(single_testing_sentence)
        '''
        train_bigram_model()
        train_trigram_model()
        '''
    

if __name__== '__main__':
    main()
