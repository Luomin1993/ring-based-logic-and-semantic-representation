#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 >>> np.cumprod(a)
...               #结果 1， 1*2=2 ，1*2*3 = 6
array([1, 2, 6])
"""


__ENV__  =  'python3';
__author =  'hanss401';

import numpy as np;
import pickle;
import make_data;
import sys

# sys.setrecursionlimit(9000000) #这里设置大一些

class Binary_Tree(object):
    """docstring for Binary_Tree"""
    def __init__(self, VALUE, LEFT, RIGHT):
        super(Binary_Tree, self).__init__()
        self.VALUE = VALUE;
        self.LEFT  = LEFT; 
        self.RIGHT =  RIGHT;

# def tree_like_space_constructing(DELTA,DEEP):
#        SPACE = [];
#        TEMP_POOL = [];
#        def tree_constructing(DELTA,ARRAY,DEEP,FIRST_FLAG):
#            # if len(ARRAY) > DEEP:return;
#            if len(ARRAY) < DEEP and FIRST_FLAG==1:print(DELTA);ARRAY.append(DELTA);
#            TEMP_POOL.append(ARRAY);
#            print(ARRAY);
#            if len(ARRAY) == DEEP:
#                SPACE.append(ARRAY);
#                return;
#            tree_constructing(abs(DELTA) ,TEMP_POOL[-1],DEEP,1);  # constructing left sub-tree;
#            tree_constructing(-abs(DELTA),TEMP_POOL[-1],DEEP,1); # constructing right sub-tree;  
#        tree_constructing(DELTA,[],DEEP,0);
#        return SPACE;    

# def tree_like_space_constructing(DELTA,DEEP):
#        SPACE = [[]];
#        def tree_constructing(DELTA,ARRAY,DEEP,FIRST_FLAG):
#            # if len(ARRAY) > DEEP:return;
#            if FIRST_FLAG==1:
#                SPACE.append(ARRAY.append(DELTA));
#            print(SPACE);
#            if len(SPACE[-1]) < DEEP:
#                tree_constructing(abs(DELTA),SPACE[-1],DEEP,1);  # constructing left sub-tree;
#                tree_constructing(-abs(DELTA),SPACE[-1],DEEP,1); # constructing right sub-tree;  
#            # if len(SPACE)==2**DEEP:return;
#        tree_constructing(DELTA,[],DEEP,0);
#        return SPACE;    

def tree_like_space_constructing(DELTA,DEEP):
    SPACE = [];
    TEMP_POOL = [];
    def tree_constructing(DELTA,ARRAY,DEEP,FIRST_FLAG):
        # if len(ARRAY) > DEEP:return;
        if len(ARRAY) < DEEP and FIRST_FLAG==1:print(DELTA);ARRAY.append(DELTA);
        TEMP_POOL.append(ARRAY);
        print(ARRAY);
        if len(ARRAY) == DEEP:
            SPACE.append(ARRAY);
            return;
        tree_constructing(abs(DELTA) ,TEMP_POOL[-1],DEEP,1);  # constructing left sub-tree;
        tree_constructing(-abs(DELTA),TEMP_POOL[-1],DEEP,1);  # constructing right sub-tree;  
    tree_constructing(DELTA,[],DEEP,0);
    return SPACE;    

def space_constructing_by_queue(DELTA,DEEP):
    SPACE = [];
    QUEUE = Queue([ [] ]);
    while (not QUEUE.empty()):
        # print(QUEUE.QUEUE);
        if len(QUEUE.top()) == DEEP:
            SPACE.append(QUEUE.pop());
            continue;
        LEFT_ARRAY = QUEUE.top(); LEFT_ARRAY = LEFT_ARRAY+ [DELTA];  # LEFT_ARRAY.append(DELTA);
        RIGHT_ARRAY = QUEUE.top(); RIGHT_ARRAY = RIGHT_ARRAY+ [-DELTA];  # RIGHT_ARRAY.append(-DELTA);
        # print(id(LEFT_ARRAY));print(id(RIGHT_ARRAY));
        QUEUE.en_queue(LEFT_ARRAY);
        QUEUE.en_queue(RIGHT_ARRAY);
        QUEUE.pop();
    return SPACE; # print(SPACE);    

class Queue(object):
    """docstring for Queue"""
    def __init__(self, QUEUE):
        super(Queue, self).__init__()
        self.QUEUE = QUEUE;

    def top(self):
        return self.QUEUE[0];

    def pop(self):
        POP = self.QUEUE[0];
        del(self.QUEUE[0])
        return POP;

    def en_queue(self,ADD_ONE):
        self.QUEUE.append(ADD_ONE);         

    def empty(self):
        return len(self.QUEUE) == 0;       

class Divide_Space(object):
    """docstring for Divide_Space"""
    def __init__(self, POINT,DIVIDE_SPACE):
        super(Divide_Space, self).__init__()
        (self.POINT_DIRECTION_SPACE_T,self.POINT_DIRECTION_SPACE_F) = deal_space(POINT,DIVIDE_SPACE);

    def deal_space(self,POINT,DIVIDE_SPACE):
        """ POINT_DIRECTION_SPACE = [... (POINT,DIRECTION) ...]  """
        POINT_DIRECTION_SPACE_T = [];  
        POINT_DIRECTION_SPACE_F = [];  
        for DIRECTION in DIVIDE_SPACE[0]:
            POINT_DIRECTION_SPACE_T.append( (POINT,DIRECTION) );
        for DIRECTION in DIVIDE_SPACE[1]:
            POINT_DIRECTION_SPACE_F.append( (POINT,DIRECTION) );    
        return (POINT_DIRECTION_SPACE_T,POINT_DIRECTION_SPACE_F);
                        

class Ring(object):
    """ Ring-based Logic """
    def __init__(self, WORD_ID,WORD,VECTOR):
        super(Ring, self).__init__()
        self.WORD_ID      =    WORD_ID;
        self.VECTOR       =     VECTOR;
        self.POLYNOMIAL   =       None;
        self.WORD         =       WORD;
        self.GRAPH        =       None;
        self.DIVIDE_SPACE =       None;
        self.POINT_DIRECTION_SPACE = None;

    def polynomial_func(self,PARAMETERS):
        if len(PARAMETERS)==2:
            RES = np.cumprod(PARAMETERS[0] - self.VECTOR)[-1]/np.cumprod(PARAMETERS[1] - self.VECTOR)[-1]; # f(x,y);
            if RES==0:
                self.deal_with_zero(PARAMETERS);
                RES = np.cumprod(PARAMETERS[0] - self.VECTOR)[-1]/np.cumprod(PARAMETERS[1] - self.VECTOR)[-1]; # f(x,y);
            return RES;
        RES = np.cumprod(PARAMETERS - self.VECTOR)[-1]; # f(x);
        if RES==0:
            self.deal_with_zero(PARAMETERS);
            RES = np.cumprod(PARAMETERS - self.VECTOR)[-1]; # f(x);
        return RES;    

    def deal_with_zero(self,PARAMETERS):
        # print(PARAMETERS)
        if len(PARAMETERS)==2:
            for i in range(PARAMETERS[0].size):
                if PARAMETERS[0][i] - self.VECTOR[i] ==0:
                    self.VECTOR[i] = PARAMETERS[0][i] - 1.0;
                if PARAMETERS[1][i] - self.VECTOR[i] ==0:
                    self.VECTOR[i] = PARAMETERS[1][i] - 1.0;    
            return;        
        for i in range(PARAMETERS.size):
            if PARAMETERS[i] - self.VECTOR[i] ==0:
                self.VECTOR[i] = PARAMETERS[i] - 1.0;

    def optimize_vector(self,TARGET,PARAMETERS):
        if len(PARAMETERS)==2:
            self.optimize_vector_two(TARGET,PARAMETERS);
            return;
        if self.polynomial_func(PARAMETERS) < TARGET: # TARGET >0 and f(x) <0;
            for i in range(len(self.VECTOR)):
                if PARAMETERS[i]-self.VECTOR[i] <0:
                    self.VECTOR[i] = PARAMETERS[i] + 1.0;
                    return;
        for i in range(len(self.VECTOR)):
            # print(self.VECTOR)
            if PARAMETERS[i]-self.VECTOR[i] >0:
                self.VECTOR[i] = PARAMETERS[i] + 1.0;
                return;

    def optimize_vector_two(self,TARGET,PARAMETERS):
        if self.polynomial_func(PARAMETERS) < TARGET: # TARGET >0 and f(x,y) <0;
            # print(self.VECTOR)
            for i in range(len(self.VECTOR)):
                if (PARAMETERS[0][i]-self.VECTOR[i])/(PARAMETERS[1][i]-self.VECTOR[i]) <0:
                    if PARAMETERS[0][i]<self.VECTOR[i]:
                        self.VECTOR[i] = PARAMETERS[0][i] - 1.0;
                        return;
                    self.VECTOR[i] = PARAMETERS[1][i] - 1.0;
                    return;
        # TARGET <0 and f(x,y) >0;                
        for i in range(len(self.VECTOR)):
            if (PARAMETERS[0][i]-self.VECTOR[i])/(PARAMETERS[1][i]-self.VECTOR[i]) >0:
                # print(self.VECTOR[i])
                self.VECTOR[i] = (PARAMETERS[0][i]+PARAMETERS[1][i])/2;
                return;                                        

    def compute_divide_space(self):
        DIM = len(self.VECTOR);
        SPACE = [];
        self.DIVIDE_SPACE = [[],[]];
        DELTA = 2.0;
        # BINARY_TREE = Binary_Tree(DELTA,0,DIM);
        # for i in range(2**DIM):
        #    SPACE.append([]);
        # SPACE = tree_like_space_constructing(DELTA,DIM);
        SPACE = np.array( space_constructing_by_queue(DELTA,DIM) );
        # print(SPACE);
        for SUB_SPACE in SPACE:
            if np.cumprod( SUB_SPACE )[-1]>0:
                self.DIVIDE_SPACE[0].append( SUB_SPACE );
                continue; 
            self.DIVIDE_SPACE[1].append(SUB_SPACE);
        # print(self.DIVIDE_SPACE);       
        self.POINT_DIRECTION_SPACE = Divide_Space(self.VECTOR,self.DIVIDE_SPACE);

    def is_subspace_of(self,DIVIDE_SPACE):
        pass;

    def print_self(self):
        print('WORD_ID:' + self.WORD_ID);
        print('WORD:' + self.WORD);
        print('VECTOR:' + str(self.VECTOR));

# ------------------- BI-Relationship ALGO FUNCTIONS ---------------------------

def intersection_set(RING_F,RING_G,POINT_DIRECTION_SPACE_F,POINT_DIRECTION_SPACE_G):
    """ f ∧ g <=>  C_f|f>0 ∩ C_g|g>0 """
    INTERSECTION_SET = [];
    for POINT_DIRECTION_F in POINT_DIRECTION_SPACE_F:
        for POINT_DIRECTION_G in POINT_DIRECTION_SPACE_G:
            if RING_F.polynomial_func(POINT_DIRECTION_G[0] + POINT_DIRECTION_G[1]) * RING_G.polynomial_func(POINT_DIRECTION_F[0] + POINT_DIRECTION_F[1]) <0:
                if RING_F.polynomial_func(POINT_DIRECTION_G[0] + POINT_DIRECTION_G[1])>0:INTERSECTION_SET.append(POINT_DIRECTION_G);
                else:INTERSECTION_SET.append(POINT_DIRECTION_F);
    return INTERSECTION_SET;            

def intersection_section(RING_F,RING_G,PARAMETERS_X,PARAMETERS_Y):
    """ f(x) ∧ g(y) <=>  f(x)>α ∧ g(y)>α """
    if RING_F.polynomial_func(PARAMETERS_X)>0 and RING_G.polynomial_func(PARAMETERS_Y)>0:
        return True;
    return False;    

def infer_established_set(RING_F,RING_G):
    """ f → g <=>  C_f|f>0 ⊆ C_g|g>0 """
    ESTABLISHED_SET = [];
    for POINT_DIRECTION_F in POINT_DIRECTION_SPACE_F:
        for POINT_DIRECTION_G in POINT_DIRECTION_SPACE_G:
            if RING_F.polynomial_func(POINT_DIRECTION_G[0] + POINT_DIRECTION_G[1]) * RING_G.polynomial_func(POINT_DIRECTION_F[0] + POINT_DIRECTION_F[1]) <0:
                if RING_G.polynomial_func(POINT_DIRECTION_F[0] + POINT_DIRECTION_F[1])>0:ESTABLISHED_SET.append(POINT_DIRECTION_F);
    return ESTABLISHED_SET;

def infer_established_section(RING_F,RING_G,PARAMETERS_X,PARAMETERS_Y,ALPHA):
    """ f(x) ∧ g(y) <=>  g(y) > f(x) > α """
    if RING_F.polynomial_func(PARAMETERS_X)>ALPHA and   RING_F.polynomial_func(PARAMETERS_X)<RING_G.polynomial_func(PARAMETERS_Y):
        return True;
    return False;    

# -------------------- TEST FUNCTIONS ----------------------
def TEST_make_a_ring(WORD_ID):
    # WORD_ID = 'ADJ___177';
    WORD    = make_data.resolve_ID(WORD_ID,'SYMBOL');
    VECTOR  = make_data.resolve_ID(WORD_ID,'VECTOR');
    RING    = Ring(WORD_ID,WORD,VECTOR);
    # RING.print_self();
    return RING;

def TEST_ring_polynomial():    
    RING_PR = TEST_make_a_ring('ADJ___173');
    RING_NN = TEST_make_a_ring('NN___84');
    print(RING_PR.polynomial_func(RING_NN.VECTOR));

def TEST_ring_polynomial():    
    RING_PR = TEST_make_a_ring('ADJ___173');
    RING_PR.compute_divide_space();

def TEST_build_bitree():
    ROOT = Binary_Tree([],0,0)
    TEST_tree_constructing(2.0,ROOT,4);


def TEST_tree_constructing(DELTA,TREE_NODE,DEEP):
    print(TREE_NODE.VALUE)
    if len(TREE_NODE.VALUE) < DEEP: 
        TREE_NODE.LEFT=Binary_Tree(TREE_NODE.VALUE,0,0);TREE_NODE.LEFT.VALUE.append( abs(DELTA));
        TREE_NODE.RIGHT=Binary_Tree(TREE_NODE.VALUE,0,0);TREE_NODE.RIGHT.VALUE.append(-abs(DELTA));
    if len(TREE_NODE.VALUE) == DEEP:
        #print(TREE_NODE.VALUE);
        return;

    TEST_tree_constructing(DELTA,TREE_NODE.LEFT,DEEP);  # constructing left sub-tree;
    TEST_tree_constructing(DELTA,TREE_NODE.RIGHT,DEEP); # constructing right sub-tree;  

# def TEST_tree_constructing(DELTA,TREE_NODE,DEEP):
#     print(TREE_NODE.VALUE)
#     if len(TREE_NODE.VALUE) < DEEP: 
#         TREE_NODE.LEFT=Binary_Tree(TREE_NODE.VALUE,0,0);TREE_NODE.LEFT.VALUE.append( abs(DELTA));
#         TREE_NODE.RIGHT=Binary_Tree(TREE_NODE.VALUE,0,0);TREE_NODE.RIGHT.VALUE.append(-abs(DELTA));
#     if len(TREE_NODE.VALUE) == DEEP:
#         #print(TREE_NODE.VALUE);
#         return;
#     TEST_tree_constructing(DELTA,TREE_NODE.LEFT,DEEP);  # constructing left sub-tree;
#     TEST_tree_constructing(DELTA,TREE_NODE.RIGHT,DEEP); # constructing right sub-tree;  


# def TEST_tree_constructing(DELTA,ARRAY,DEEP,ARR_ID):
#     if len(ARRAY) < DEEP: 
#         LEFT_ARRAY=ARRAY;LEFT_ARRAY.append( abs(DELTA));
#         RIGHT_ARRAY=ARRAY;RIGHT_ARRAY.append(-abs(DELTA));
#     if len(ARRAY) == DEEP:
#         print(ARRAY);
#         return;
#     TEST_tree_constructing(DELTA,LEFT_ARRAY,DEEP,ARR_ID);  # constructing left sub-tree;
#     TEST_tree_constructing(DELTA,RIGHT_ARRAY,DEEP,ARR_ID+1); # constructing right sub-tree;  

def TEST_optimize():
    X1 = np.array([1.,2.,3.,4.]);
    X2 = np.array([3.,6.,11.,14.]);
    WORD_ID = 'ADJ___177';
    WORD    = make_data.resolve_ID(WORD_ID,'SYMBOL');
    VECTOR  = make_data.resolve_ID(WORD_ID,'VECTOR');
    RING    = Ring(WORD_ID,WORD,VECTOR);
    print(RING.polynomial_func([X1,X2]));
    RING.optimize_vector(.0001,[X1,X2]);
    print(RING.polynomial_func([X1,X2]));

    print(RING.polynomial_func(X1));
    RING.optimize_vector(-0.0001,X1);
    print(RING.polynomial_func(X1));

if __name__ == '__main__':
    # TEST_make_a_ring(); 
    # TEST_ring_polynomial();           
    # TEST_ring_polynomial();
    # TEST_tree_constructing(2.0,[],4,0);
    # TEST_build_bitree();
    # TEST_space_constructing_by_queue(2.0,4);
    TEST_optimize();