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

def pre_process_data():        

        pre_processing = prep.PreProcessing() 
        padded_train_data = pre_processing.pad_file_data('C:/Users/Evgeniia/Desktop/GitHub Project Folder/Python Projects/N-Grams/a.txt')
        train_dict = pre_processing.create_dictionary(padded_train_data)
        processed_train_data = pre_processing.mark_unknown_words(1, padded_train_data, train_dict)
        processed_train_dict = pre_processing.create_dictionary(processed_train_data)
        
        padded_test_data = pre_processing.pad_file_data('C:/Users/Evgeniia/Desktop/GitHub Project Folder/Python Projects/N-Grams/b.txt')
        test_dict = pre_processing.create_dictionary(padded_test_data)
        processed_test_data = pre_processing.mark_unknown_words(2, padded_test_data, processed_train_dict)
        processed_test_dict = pre_processing.create_dictionary(processed_test_data)
        
        return padded_train_data, train_dict, processed_train_data, processed_train_dict, padded_test_data, test_dict, processed_test_data, processed_test_dict
    
    
    
    
    

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
