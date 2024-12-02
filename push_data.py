import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi # provides set of root certificate
ca=certifi.where() # ca = cerficate authority

import pandas as pd
import numpy as np
import pymongo
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException

class NetworkDataExtract():
    def __init__(self):
        try:
            # self.database=database, can write here also
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def cvs_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insrt_data_mongodb(self, records, database, collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records
            
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
if __name__=='__main__':
    FILE_PATH="network_data\phisingData.csv"
    DATABASE="JAYYL"
    Collection="NetworkData"
    networkobj=NetworkDataExtract()
    records=networkobj.cvs_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=networkobj.insrt_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)