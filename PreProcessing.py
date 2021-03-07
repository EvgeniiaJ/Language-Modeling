class PreProcessing:
    
    def __init__(self):
        pass
    
    # make all letters of the entire document lowercase
    # pad each distinct sentence in the data set with start (<s>) 
    #       and end (</s>) symbols
    def pad_file_data(self, input_file):
        
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
    
    def create_dictionary(self, padded_data):
        word_dict = dict()
        
        for word in padded_data:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
                
        return word_dict
    
    
    # mark words with unknown symbol (<unk>) if they are seen no more than once
    def mark_unknown_words(self, pass_id, data, file_dict):
        
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
        
        
