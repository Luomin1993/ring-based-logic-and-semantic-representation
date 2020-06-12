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
    def __init__(self, DELTA, ARRAY, DEEP):
        super(Binary_Tree, self).__init__()
        self.DELTA = DELTA;
        if len(LAYER)<DEEP:
            self.LEFT  = Binary_Tree(abs(DELTA),LAYER+1,DEEP);
            self.RIGHT = Binary_Tree(-abs(DELTA),LAYER+1,DEEP);    

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
#    	SPACE = [[]];
#    	def tree_constructing(DELTA,ARRAY,DEEP,FIRST_FLAG):
#    	    # if len(ARRAY) > DEEP:return;
#    	    if FIRST_FLAG==1:
#    	        SPACE.append(ARRAY.append(DELTA));
#    	    print(SPACE);
#    	    if len(SPACE[-1]) < DEEP:
#    	        tree_constructing(abs(DELTA),SPACE[-1],DEEP,1);  # constructing left sub-tree;
#    	        tree_constructing(-abs(DELTA),SPACE[-1],DEEP,1); # constructing right sub-tree;  
#    	    # if len(SPACE)==2**DEEP:return;
#    	tree_constructing(DELTA,[],DEEP,0);
#    	return SPACE;    

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
           tree_constructing(-abs(DELTA),TEMP_POOL[-1],DEEP,1); # constructing right sub-tree;  
       tree_constructing(DELTA,[],DEEP,0);
       return SPACE;    



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

    def polynomial_func(self,PARAMETERS):
        return np.cumprod(PARAMETERS - self.VECTOR)[-1];

    def optimize_vector(self,TARGET,PARAMETERS):
        pass;

    def compute_divide_space(self):
        DIM = len(self.VECTOR);
        SPACE = [];
        self.DIVIDE_SPACE = [];
        DELTA = 2.0;
        # BINARY_TREE = Binary_Tree(DELTA,0,DIM);
        # for i in range(2**DIM):
        #    SPACE.append([]);
        SPACE = tree_like_space_constructing(DELTA,DIM);
        print(SPACE);

    def is_subspace_of(self,DIVIDE_SPACE):
        pass;

    def print_self(self):
        print('WORD_ID:' + self.WORD_ID);
        print('WORD:' + self.WORD);
        print('VECTOR:' + str(self.VECTOR));

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

def TEST_tree_constructing(DELTA,ARRAY,DEEP,FIRST_FLAG):
    if len(ARRAY) < DEEP: 
        LEFT_ARRAY=ARRAY;LEFT_ARRAY.append( abs(DELTA));
        RIGHT_ARRAY=ARRAY;RIGHT_ARRAY.append(-abs(DELTA));
    if len(ARRAY) == DEEP:
        print(ARRAY);
        return;
    THIS_ARRAY = ARRAY;    
    TEST_tree_constructing(DELTA,LEFT_ARRAY,DEEP,1);  # constructing left sub-tree;
    TEST_tree_constructing(DELTA,RIGHT_ARRAY,DEEP,1); # constructing right sub-tree;  


if __name__ == '__main__':
    # TEST_make_a_ring(); 
    # TEST_ring_polynomial();           
    # TEST_ring_polynomial();
    TEST_tree_constructing(2.0,[],4,0);