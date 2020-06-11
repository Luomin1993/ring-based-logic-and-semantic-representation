#!/usr/bin/env python
# -*- coding: utf-8 -*-

__ENV__  =  'python3';
__author =  'hanss401';

import numpy as np;
import pickle;

TRUE_VALUE_LINE = 20.0;
WORD_PROPERTY_DICT = {'NN' :['NN'],
                      'PR' :['VBG','VBZ'],
                      'ADJ':['JJ'],
                      'ADV':['RB']};
NUM_LIM_PR = 2;
NUM_LIM_NN = 3;
NUM_LIM_ADJ = 2;
NUM_LIM_ADV = 1;

SAMPLES_EXAMPLE = ({'PR_122':['NN_13','NN_72'],'ADJ_22':'NN_13','ADJ_67':'NN_72','ADV_187':'PR_122'} ,  {'PR_145':['NN_33','NN_57'],'ADJ_22':'NN_57'});

def make_random_vec(LOW = -5.0,HIGH = 5.0,DIM = 10):
    return np.random.rand(DIM)*(HIGH - LOW) + LOW;

def make_word_vec_set(WORD_PROPERTY,WORD_NUM):
    WORD_VEC_SET = [];
    for i in range(WORD_NUM):
        WORD_VEC_SET.append( make_random_vec() );
    np.save( WORD_PROPERTY,np.array(WORD_VEC_SET) );

def make_word_symbol_set(WORD_PROPERTY,WORD_NUM):
    WORD_SYMBOL_SET = [];
    with open('snli_1.0/snli_1.0_test.txt','r') as FILE_STREAM:
        SNLI_CONTENT = FILE_STREAM.read();
        SNLI_CONTENT = SNLI_CONTENT.replace('\n','').replace('(','').replace(')','').split(' ');
    for i in range(len(SNLI_CONTENT)):
        if SNLI_CONTENT[i] in WORD_PROPERTY_DICT[WORD_PROPERTY]:
            # if SNLI_CONTENT[i+1] != '(' and SNLI_CONTENT[i+1] != ')':
            if (SNLI_CONTENT[i+1] not in WORD_SYMBOL_SET) and ('	' not in SNLI_CONTENT[i+1]):
                WORD_SYMBOL_SET.append(SNLI_CONTENT[i+1]);
        if len(WORD_SYMBOL_SET) > WORD_NUM:break;
    with open(WORD_PROPERTY+'.dst','w') as FILE_STREAM:
        for i in range(WORD_NUM):
            # WORD_VEC_SET.append( THIS_WORD_SYMBOL + '    ' + THIS_WORD_PROPERTY);
            # FILE_STREAM.write(THIS_WORD_SYMBOL + '    ' + THIS_WORD_PROPERTY);    
            FILE_STREAM.write(WORD_SYMBOL_SET[i]);    
            FILE_STREAM.write('\n');

def make_expirience_set(SAMPLES_NUM):
    EXPIRIENCE_SET = [];
    SAMPLES_SET    = [];
    for i in range(SAMPLES_NUM*2):
        NUM_NN  = np.random.randint(low = 1,high = NUM_LIM_NN); NN_SET = [];
        NUM_PR  = np.random.randint(low = 1,high = NUM_LIM_PR); PR_SET = [];
        NUM_ADV = np.random.randint(low = 0,high = NUM_LIM_ADV); ADV_SET = [];
        NUM_ADJ = np.random.randint(low = 0,high = NUM_LIM_ADJ); ADJ_SET = [];
        for i in range(NUM_NN):NN_SET.append(   'NN'   +'___'+ str(np.random.randint(low=0,high=1000))  );
        for i in range(NUM_PR):PR_SET.append(   'PR'   +'___'+ str(np.random.randint(low=0,high=300))   );
        for i in range(NUM_ADJ):ADJ_SET.append(  'ADJ' +'___'+ str(np.random.randint(low=0,high=500))   );
        for i in range(NUM_ADV):ADV_SET.append(  'ADV' +'___'+ str(np.random.randint(low=0,high=300))   );
        # make a sample:
        THIS_SAMPLE = {};
        for PREDICATE in PR_SET:
            if NUM_NN==1:THIS_SAMPLE[PREDICATE] = NN_SET[0];
            if NUM_NN>1:THIS_SAMPLE[PREDICATE] = [NN_SET[0],NN_SET[1]];
            if NUM_PR>1 and PREDICATE==PR_SET[-1]:THIS_SAMPLE[PREDICATE] = NN_SET[-1];
        if NUM_ADV>0:THIS_SAMPLE[ADV_SET[0]] = PR_SET[0];
        if NUM_ADJ>0:
            if NUM_ADJ >  NUM_NN:THIS_SAMPLE[ADJ_SET[0]] = NN_SET[0];
            if NUM_ADJ <= NUM_NN:
                for i in range(len(ADJ_SET)):THIS_SAMPLE[ADJ_SET[i]] = NN_SET[i];
        SAMPLES_SET.append(THIS_SAMPLE);    
    for i in range(SAMPLES_NUM):
        EXPIRIENCE_SET.append(  ( SAMPLES_SET[2*i] , SAMPLES_SET[2*i+1] )  );
    SAVED_FILE = open('expirience_set.dst', 'wb');    
    pickle.dump(EXPIRIENCE_SET, SAVED_FILE);            


# -------------------- TEST FUNCTIONS ----------------------
def TEST_make_word_symbol_set():
    make_word_symbol_set('ADV',200);

def TEST_make_expirience_set():
    make_expirience_set(20);    


if __name__ == '__main__':
    # TEST_make_word_symbol_set();
    TEST_make_expirience_set();                