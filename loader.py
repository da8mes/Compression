# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import torch.utils.data as data
from torch import from_numpy
import h5py
import numpy as np
# import concurrent.futures
import threading 
from multiprocessing import Pool, Manager, Process
from multiprocessing.sharedctypes import Value
from collections import ChainMap
import time
from timeit import default_timer as timer

def load_files(pdgIDs, filters, dataname):
    data = {}
    file_data = load_hdf5(dataname, pdgIDs)
    # apply any filters here
    if not filters is None:
        for filt in filters: filt.filter(file_data)
    for key in file_data.keys(): 
        if key in data.keys():
            data[key] = np.append(data[key], file_data[key], axis=0)
        else:
            data[key] = file_data[key]
    return data 

def load_ECAL(file, index):
    res = np.empty((500, 51, 51, 25))
    f['ECAL'].read_direct(res, np.s_[index-500:index])
    return res

# no speed up when using parallel because it's not truely parallel. 
def pload_hdf5(file, pdgIDs, loadMinimalFeatures=None):
    '''Loads H5 file. Used by HDF5Dataset.'''
    return_data = {}
    ind = range(500, 10001, 500)
    pool = Pool(processes=20)
    res = pool.starmap(load_ECAL, [(file, x) for x in ind])
    return_data['ECAL'] = np.concatenate(res)
    with h5py.File(file, 'r') as f:
        # (default) load full ECAL / HCAL arrays and standard features
        if loadMinimalFeatures is None: 
            n_events = 10000
            return_data['HCAL'] = f['HCAL'][:].astype(np.float32)
            return_data['pdgID'] = f['pdgID'][:].astype(int)
            return_data['classID'] = np.array([pdgIDs[abs(i)] for i in return_data['pdgID']]) # PyTorch expects class index instead of one-hot
            
            other_features = ['ECAL_E', 'HCAL_E', 'HCAL_ECAL_ERatio', 'energy', 'eta', 'recoEta', 'phi', 'recoPhi', 'openingAngle']
            for feat in other_features:
                if feat in f.keys(): return_data[feat] = f[feat][:].astype(np.float32)
                else: return_data[feat] = np.zeros(n_events, dtype=np.float32)
        # minimal data load: only load specific features that are requested
        else:
            for feat in loadMinimalFeatures:
                return_data[feat] = f[feat][:]
    return return_data

def load_hdf5(file, pdgIDs, loadMinimalFeatures=None):
    '''Loads H5 file. Used by HDF5Dataset.'''
    return_data = {}
    with h5py.File(file, 'r') as f:
        # (default) load full ECAL / HCAL arrays and standard features
        if loadMinimalFeatures is None:
#             return_data['ECAL'] = np.empty((10000, 51, 51, 25), dtype='float32')
#             f['ECAL'].read_direct(return_data['ECAL'])
            return_data['ECAL'] = f['ECAL'][:].astype(np.float32)
#             print('ECAL')
            n_events = 10000 #len(return_data['ECAL'])
            return_data['HCAL'] = f['HCAL'][:].astype(np.float32)
#             return_data['HCAL'] = np.empty(f['HCAL'].shape, dtype='float32')
#             f['HCAL'].read_direct(return_data['HCAL'])
            return_data['pdgID'] = f['pdgID'][:].astype(np.int)
#             return_data['pdgID'] = np.empty(f['pdgID'].shape, dtype='int')
#             f['pdgID'].read_direct(return_data['pdgID'])
            return_data['classID'] = np.array([pdgIDs[abs(i)] for i in return_data['pdgID']]) # PyTorch expects class index instead of one-hot
            
            other_features = ['ECAL_E', 'HCAL_E', 'HCAL_ECAL_ERatio', 'energy', 'eta', 'recoEta', 'phi', 'recoPhi', 'openingAngle']
            for feat in other_features:
                if feat in f.keys(): return_data[feat] = f[feat][:].astype(np.float32)
                else: return_data[feat] = np.zeros(n_events, dtype=np.float32)
        # minimal data load: only load specific features that are requested
        else:
            for feat in loadMinimalFeatures:
                return_data[feat] = f[feat][:]
    return return_data

class HDF5Dataset(data.Dataset):

    """Creates a dataset from a set of H5 files.
        Used to create PyTorch DataLoader.
    Arguments:
        dataname_tuples: list of filename tuples, where each tuple will be mixed into a single file
        num_per_file: number of events in each data file
    """

    def __init__(self, dataname_tuples, pdgIDs, filters=[]):
        self.dataname_tuples = sorted(dataname_tuples)
        self.nClasses = len(dataname_tuples[0])
        self.total_files = len(dataname_tuples) # per class
        self.num_per_file = len(dataname_tuples) * [0]
        self.fileInMemory = Value('i', -1)
        self.fileInMemoryFirstIndex = Value('i', 0)
        self.fileInMemoryLastIndex = Value('i', -1)
        self.loadNext = Value('i', 1)
        self.data = {}
        self.pdgIDs = {}
        self.filters = filters
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i
        self.countEvents()

    def countEvents(self):
        # make minimal list of inputs needed to check (or count events)
        minFeatures = []
        for filt in self.filters: minFeatures += filt.featuresUsed
        # use energy to count number of events if no filters
        if len(minFeatures) == 0: minFeatures.append('energy')
        totalevents = 0
        # num_per_file and totalevents count the minimum events in a file tuple (one file for each class)
        for fileN in range(len(self.dataname_tuples)):
            nevents_before_filtering, nevents_after_filtering = [], []
            for dataname in self.dataname_tuples[fileN]:
                file_data = load_hdf5(dataname, self.pdgIDs, minFeatures)
                nevents_before_filtering.append(len(list(file_data.values())[0]))
                for filt in self.filters: filt.filter(file_data)
                nevents_after_filtering.append(len(list(file_data.values())[0]))
            self.num_per_file[fileN] = min(nevents_after_filtering) * self.nClasses
            totalevents += min(nevents_before_filtering) * self.nClasses
        print('total events:',totalevents)
        if len(self.filters) > 0:
            print('total events passing filters:',sum(self.num_per_file))
            
    def load_file(self): 
        # update indices to new file
        print("loading file")
        self.fileInMemory.value += 1
        self.fileInMemoryFirstIndex.value = int(self.fileInMemoryLastIndex.value+1)
        self.fileInMemoryLastIndex.value += self.num_per_file[self.fileInMemory.value]
        # print(index, self.fileInMemory, self.fileInMemoryFirstIndex, self.fileInMemoryLastIndex)
        self.data = {}
        for dataname in self.dataname_tuples[self.fileInMemory.value]: # most time consuming I/O
            file_data = load_hdf5(dataname, self.pdgIDs)
            # apply any filters here
            if not self.filters is None:
                for filt in self.filters: filt.filter(file_data)
            for key in file_data.keys():
                if key in self.data.keys():
                    self.data[key] = np.append(self.data[key], file_data[key], axis=0)
                else:
                    self.data[key] = file_data[key]
    
    def _init_worker(self, worker_id):
        if worker_id == 0:
            # periodically check if file needs to be loaded
            # using fileLastIndex value 
            # when we run out of events, load files.
            # communicate with other process using the wait flag
            # if wait flag is True, other processes will stop and wait for this 
            # worker to finish loading the next files.
            # if wait flag is False, processes will continue with 
            # normal work. 
            # if all files needed: terminate process
            # reserve worker 0 for file loading only 
            
            #TODO 
            # use idle workers to load files simultaneously.
            while (self.fileInMemory.value + 1) < self.total_files: 
#                 print("worker 0: waiting to load")
                # periodically check if file needs to be loaded
                # only break when all files are loaded 
                if self.loadNext:
                    print("--------------------")
                    print("I'm loading next file")
                    print(self.fileInMemory.value)
                    self.load_file()
                    print("I'm done loading next file")
                    print("---------------------")
                    self.loadNext.value = 0
        
#     def worker_init_fn(worker_id): #currying _init_workers 
#         return self._init_worker(worker_id)

    def __getitem__(self, index): # if you get here you are not worker 0 (loader) 
        # if entering a new epoch, re-initialze necessary variables
        if (index < self.fileInMemoryFirstIndex.value): 
            self.fileInMemory.value = -1
            self.fileInMemoryFirstIndex.value = 0
            self.fileInMemoryLastIndex.value = -1
        # if we started to look at a new file, read the file data
        if(index > self.fileInMemoryLastIndex.value):
            self.loadNext.value = 1 
            # wait till next file is loaded by worker 0 before threading on batch.  
            while self.loadNext.value:
                print("waiting till next file is loaded.")
                time.sleep(10)  # might want to optimize sleep time for threads. 
    
        # return the correct sample
        indexInFile = index - self.fileInMemoryFirstIndex.value
        return_data = {} #pytorch multiprocessing doesn't have significant advantage because io is bound by the 
        # previous for loop which is not prallelized and also the overhead is more than the speed up one gets because 
        # rertunrning a section of self.data is not as costy as loading an entire file and spawning threads. 
        # For pytorch's parallel loading to produce speed up we need a lot of workers but we are memory bound. 
        for key in self.data.keys(): 
            return_data[key] = self.data[key][indexInFile]
        return return_data # returns sections of ECAL and other attributes event by event. 

    def __len__(self):
        return sum(self.num_per_file)

class OrderedRandomSampler(object):

    """Samples subset of elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.num_per_file = self.data_source.num_per_file

    def __iter__(self):
        indices=np.array([],dtype=np.int64)
        prev_file_end = 0
        for i in range(len(self.num_per_file)):
            indices=np.append(indices, np.random.permutation(self.num_per_file[i])+prev_file_end)
            prev_file_end += self.num_per_file[i]
        return iter(from_numpy(indices))

    def __len__(self):
        return len(sum(self.num_per_file))
    
    
