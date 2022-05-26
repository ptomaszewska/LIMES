import json
import pandas as pd
import numpy as np
import os
import sys
import glob
import sentence_transformers
from sentence_transformers import SentenceTransformer

"""
The input to the algorithm are the files with extracted information from raw Twitter data. 
Another commandline argument is the path where the files ready to be used for training are to be stored. 
The goal is to embed the text of the tweet and the location meta data using pretrained sentence transformer.
In the script the country_codes.csv file (provided in the repostory) is used.
At this stage random split into training and validation sets is done. 
"""

def create_embeddings(filename, save_path):
    with open(filename,'r') as json_file:
        json_list=list(json_file)

    countries_val=[]
    countries_train=[]
    counter_train=0
    counter_val=0
    for json_str in json_list:
        result=json.loads(json_str)
         
        if result['country'] in dict_code['it'] and result['location'] is not None:
            country=dict_code['it'][result['country']]
            
            sentence_embedding=model.encode(result['text'])
            location_embedding=model.encode(result['location'])
                
            p=np.random.choice([0,1], size=1, p=[0.8,0.2])[0]
            features=np.concatenate((sentence_embedding, location_embedding), axis=0)
            if p==0: #training set
                countries_train.append(country)
                if counter_train==0:
                    embs_train=features.reshape(1,-1)
                else:
                    embs_train=np.append(embs_train,features.reshape(1,-1), axis=0)
                counter_train+=1
            else: #validation set
                if counter_val==0:
                    embs_val=features.reshape(1,-1)
                else:
                    embs_val=np.append(embs_val,features.reshape(1,-1), axis=0)
                counter_val+=1
                countries_val.append(country)

        else:
            continue
    
    data_train=np.concatenate((embs_train, np.array(countries_train).reshape(-1,1)), axis=1)
    data_train=data_train.astype('float32')
    data_val=np.concatenate((embs_val, np.array(countries_val).reshape(-1,1)), axis=1)
    data_val=data_val.astype('float32')
    
    path_train=os.path.join(save_path, "train")
    path_val=os.path.join(save_path, "val")
    if not os.path.exists(path_train):
          os.makedirs(path_train)
    if not os.path.exists(path_val):
        os.makedirs(path_val)
    np.save(path_train"+str(filename[-22:-9]),data_train)
    np.save(path_val+str(filename[-22:-9]),data_val)

if __name__=="__main__":
    model=SentenceTransformer('distiluse-base-multilingual-cased-v1') ## suitable for social media texts

    np.random.seed(1)

    country_code=pd.read_csv('country_codes.csv', header=None, delimiter=';', encoding='latin')
    country_code.columns=['it','country','code']
    country_code.drop('country',axis=1, inplace=True)

    dict_code=country_code.set_index('code').to_dict()
    
    filename=sys.argv[1]
    save_path=sys.argv[2]

    create_embeddings(filename, save_path)

