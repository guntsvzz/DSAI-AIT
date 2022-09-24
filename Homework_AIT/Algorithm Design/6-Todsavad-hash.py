'''
Here I gonna perform chaining method

Assignment:  Linear Probing
'''
from asyncio.windows_events import NULL


class HashTable:
    
    def __init__(self, m):
        self.m = m
        self.hashtable = self.create_hash_table()
    
    def create_hash_table(self):
        return [ [] for _ in range(self.m) ]
    
    def _prehash(self, key):
        #challenge: handle negative keys and string
        if (type(key) == str):
            key = hash(key)  #returns a number for you
            
        if ((type(key) == int) | (type(key) == float)):
            if (key < 0):
                key = hash(float(key)) * -1  #first convert to float, then hash it
        
        assert (key > 0) & (type(key) == int)
    
        return key
    
    def _hash(self, key):
        #get the position using division method
        index = key % self.m
        bucket = self.hashtable[index]
        return bucket
    
    def insert(self, key, val):
        key    = self._prehash(key)  #clean neg numbers or string
        bucket = self._hash(key)     #get the position of the hashtable
        
        # found = False
        # #check whether the key duplicates
        # for i, (bkey, bval) in enumerate(bucket):
        #     if bkey == key:
        #         found = True
        #         pos_dup = i
        #         break
        found, pos_dup, _ = self.search(key)
                
        #if the key duplicates, only update the value
        if(found):
            bucket[pos_dup] = (key, val)
            #Probing 
        else: #if the key does not exist, append and #if something is there already, append
            bucket.append((key, val))
            
        print(self.hashtable)
    
    def search(self, key):
        #if you finish this, 
        key = self._prehash(key)
        
        #perform the division method
        bucket = self._hash(key)     #get the position of the hashtable

        found  = False
        answer = -9999
        pos_dup = -9999
        #loop the bucket index
        for i, (bkey, bval) in enumerate(bucket):
            if bkey == key:
                found   = True
                pos_dup = i
                answer  = bval
                break
        return found, pos_dup, answer  
    
    def delete(self, key):
        #implement this too
        key    = self._prehash(key)
        bucket = self._hash(key)
        found, _, _ = self.search(key)
        if(found):
            bucket.clear()
        print(self.hashtable)
    
ht = HashTable(11)
ht.insert(1, 'Chaky')
ht.insert(2, 'Peter')
ht.insert(2, 'Peterss')
ht.insert(3, 'John')
ht.insert(12, 'Matthew')  #this should be in the same bucket with 'Chaky'
ht.delete(2)

# found, _, val = ht.search(12)
# print(found, val)