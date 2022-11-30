from gui import mainScreen
import spacy
from spacy import displacy

def convertToSpacy(text):
    nlp=spacy.load('en_core_web_sm/en_core_web_sm-3.4.1/')
    # text= 'book the flight through houston' #arc-eager
    # text = 'I am hungry' #arc-standard
    # text = 'Viken will join the board as a nonexecutive director Nov 29'
    word_list = list()
    # dep_list = list()
    for index,token in enumerate(nlp(text)):
        print(index+1, token.text,'=>',token.dep_,'=>',token.head.text)
        # text = np.array([index+1,token.text,token.dep_])
        # np.append(list_text,text)
        word_list.append((index+1,token.text,token.head.text,token.dep_)) #token.dep_ 
        # dep_list.append((token.text,token.dep_,token.head.text))
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

def convertToList(new_config):
    #store to list
    from collections import defaultdict
    graph = defaultdict(list)
    for i in new_config.arcs:
        graph[f"{i[0][0]}"].append(f"{i[2][0]}")
    print('Original graph :\n',graph)
    graph_sort = dict(sorted(graph.items()))
    print('Sorted graph :\n', graph_sort)
    return graph

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


###############################################################################################3