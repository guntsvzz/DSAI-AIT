import spacy
from spacy import displacy
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from timeit import timeit

###############################################################################################

import spacy
from spacy import displacy
def convertToSpacy(text):
    nlp=spacy.load('en_core_web_sm/en_core_web_sm-3.4.1/')
    word_list = list()
    print ("{:<5} | {:<10} | {:<8} | {:<10} | {:<20}".format('Index','Token','Relation','Head', 'Children'))
    print ("-" * 70)
    for index,token in enumerate(nlp(text)):
        print ("{:<5} | {:<10} | {:<8} | {:<10} | {:<20}"
         .format(str(index+1),str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))
        word_list.append((index+1,token.text,token.head.text,token.dep_)) #token.dep_ 
    displacy.render(nlp(text),jupyter=True)
      
    return word_list

###############################################################################################

class Configuration(object):
    def __init__(self,dependency_list):
        self.stack = [(0,'root','root','ROOT')]
        self.buffer = dependency_list
        self.arcs = list()
    def __str__(self):
        return f'Stack  : {self.stack} \nBuffer : {self.buffer} \nArcs   : {self.arcs}'

class Transition(object): ####put word to buffer####
    def __init__(self,approach):
        self.approach = approach #'arc-standard' 'arc-eager'
    #Arc-standard parsing cannot produce non-projective trees
    def left_arc(self,config,relation):
        if self.approach == 'arc-standard':
            #pop top of stack -> append arc relation
            index_i = config.stack.pop()
            index_j = config.stack.pop()
            config.stack.append(index_i)
            config.arcs.append((index_i, relation, index_j)) 
        elif self.approach == 'arc-eager':
            pass

    def right_arc(self,config,relation):
        if self.approach == 'arc-standard':
            #pop top of stack -> append arc relation
            index_i = config.stack.pop()
            index_j = config.stack.pop()
            config.stack.append(index_j)
            config.arcs.append((index_j, relation, index_i)) 
        elif self.approach == 'arc-eager':
            pass

    def shift(self,config): #move buffer to stack
        if len(config.buffer) <= 0:
            return -1
        index_i = config.buffer.pop(0)
        config.stack.append(index_i)

    def reduce(sefl,config):
        pass

class Parser(object): 
    def __init__(self,approach):
        self.approach = approach

    def oracle(self,config): ####put buffer to stack####
        operation = Transition(self.approach)
        i = 0
        print('Action :',end=' ')
        while not(len(config.buffer) == 0 and len(config.stack) == 1): #stop when buffer is empty and stack contain only root
            if (len(config.buffer) == 0 and len(config.stack) == 2):
                print('Right-Arc',end='->')
                operation.right_arc(config,'->')
                
            if len(config.stack) == 1:
                print("Shift_along",end='->')
                operation.shift(config)
            else:
                if config.stack[-1][1] == config.stack[-2][2]: 
                    print('Left-Arc',end='->')
                    operation.left_arc(config,'->')
                elif (config.stack[-1][2] == config.stack[-2][1]) : #next_head = prev_text
                    print('Right-Arc',end='->')
                    operation.right_arc(config,'->')
                else:
                    print("Shift",end='->')
                    operation.shift(config)
            i+=1
            if i == 20:
                break
        # print('\n',config)
        print('\n') 
        return config

def SaB(word_list):
    print('Setup Configuration')
    buffer_list = word_list.copy()
    config = Configuration(buffer_list)
    print(config)
    print('Trainsition-Based')
    parsing = Parser('arc-standard')
    new_config = parsing.oracle(config)
    print(new_config)
    return new_config

def DependencyGraph(word_list,new_config):
    from collections import defaultdict
    graph_default = defaultdict(list)
    for i in range(len(word_list)+1):
        # for j in range(len(word_list)+1):
        # graph_default.setdefault(i)
        graph_default[f'{i}'] = []
    for i in new_config.arcs:
    #     graph_default[i[0][0]][i[2][0]] = ((i[2][0],i[2][0]))
        graph_default[f"{i[0][0]}"].append(f'{i[2][0]}')
    return graph_default

###############################################################################################3

from collections import deque
class Stack(object):
    def __init__(self):
        self.container = deque()
    def push(self,value):
        self.container.append(value)
    def pop(self):
        if len(self.container) != 0:
            return self.container.pop()  
        else: raise IndexError("An empty deque")
    def size(self):
        return len(self.container)

def BFS(graph,s):
    visited = set() #unique number
    queue = set()   #list in python is basiaclly queue
    visited.add(s)  #means make it black
    queue.add(s)

    while queue:        #as long as the queue is not empty....
        u = queue.pop() #pop the front guy.... basiaclly index 0
        
        print(u, "-->", end = " ") 
        for neighbor in graph[u]:       #for everyone who connects to u,
            if neighbor not in visited:
                visited.add(neighbor)    #add them to visited
                queue.add(neighbor)      #add them to the queue

visited = set()
def DFS(graph,s):
    if s not in visited:
        print(s,'-->',end=" ")
        visited.add(s)
        for neighbor in graph[s]:
            DFS(graph,neighbor)


###############################################################################################

def comparision(word_list,s,graph):
    num_samples = range(len(word_list))
    c = 1/500000
    time_interval_Stack = []
    time_interval_BFS = []
    time_interval_DFS = []

    for n in num_samples:
        time1 = timeit('s.push(num_samples)',number=1,globals=globals())
        time_interval_Stack.append(time1)
        time2 = timeit('BFS(graph,str(0))',number=1,globals=globals())
        time_interval_BFS.append(time2)
        time3 = timeit('DFS(graph,str(0))',number=1,globals=globals())
        time_interval_DFS.append(time3)

    plt.plot(num_samples,time_interval_Stack,label='Stack')
    plt.plot(num_samples,time_interval_BFS,label='BFS')
    plt.plot(num_samples,time_interval_DFS,label='DFS')
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Time Complexity")
    plt.title("Time Complexity Comparison")
    plt.show()