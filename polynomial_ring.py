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

# ============ PUBLIC DATA ==============

RUNNING_DATA = {};
ALPHA        = 10.00;

def read_data():
    """Read the static data into this;"""
    ADV_MAT = np.load('ADV.npy');
    ADJ_MAT = np.load('ADJ.npy');
    PR_MAT = np.load('PR.npy');    
    NN_MAT = np.load('NN.npy');
    for i in range(ADV_MAT.shape[0]):RUNNING_DATA['ADV___'+str(i)] = ADV_MAT[i];
    for i in range(ADJ_MAT.shape[0]):RUNNING_DATA['ADJ___'+str(i)] = ADJ_MAT[i];
    for i in range(PR_MAT.shape[0]):RUNNING_DATA['PR___'+str(i)] = PR_MAT[i];
    for i in range(NN_MAT.shape[0]):RUNNING_DATA['NN___'+str(i)] = NN_MAT[i];

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


class Logic_Proposition(object):
    """docstring for Logic_Proposition"""
    def __init__(self, DICT_PAIR):
        super(Logic_Proposition, self).__init__()
        self.DICT_PAIR  = DICT_PAIR;
        self.CONDITIONS = [];
        self.CONCLUSIONS = [];
        self.RING_PARA_PAIR_CD = [];  # 条件...
        self.RING_PARA_PAIR_CC = [];  # 结论...
        self.compute_predicate_values();

    def compute_predicate_values(self):
        for PREDICATE,NOUN in self.DICT_PAIR[0].items():
            THIS_WORD = make_data.resolve_ID(PREDICATE,'SYMBOL');
            THIS_ID   = PREDICATE;
            THIS_VEC  = RUNNING_DATA[PREDICATE];
            if type(NOUN)==list: THIS_PARA = [ RUNNING_DATA[NOUN[i]] for i in range(2)];
            else: THIS_PARA = RUNNING_DATA[NOUN];
            THIS_RING = Ring(THIS_ID,THIS_WORD,THIS_VEC);
            # 执行optimize_vector;使谓词逻辑为真:
            THIS_ALPHA = THIS_RING.polynomial_func(THIS_PARA);
            if THIS_ALPHA<0:
                THIS_RING.optimize_vector(.0001,THIS_PARA);
                RUNNING_DATA[THIS_RING.WORD_ID] = THIS_RING.VECTOR; 
            THIS_ALPHA = THIS_RING.polynomial_func(THIS_PARA); # 更新ALPHA;
            self.RING_PARA_PAIR_CD.append( [ THIS_RING,THIS_PARA, THIS_ALPHA,ALPHA  ] ); # ALPHA是期望调整值 -5代表期望小于5 5代表期望大于5;
            RUNNING_DATA[THIS_RING.WORD_ID] = THIS_RING.VECTOR;  # 更新全局变量;
        for PREDICATE,NOUN in self.DICT_PAIR[1].items():
            THIS_WORD = make_data.resolve_ID(PREDICATE,'SYMBOL');
            THIS_ID   = PREDICATE;
            THIS_VEC  = RUNNING_DATA[PREDICATE];
            if type(NOUN)==list: THIS_PARA = [ RUNNING_DATA[NOUN[i]] for i in range(2)];
            else: THIS_PARA = RUNNING_DATA[NOUN];
            THIS_RING = Ring(THIS_ID,THIS_WORD,THIS_VEC);
            # 执行optimize_vector;使谓词逻辑为真:
            THIS_ALPHA = THIS_RING.polynomial_func(THIS_PARA);
            if THIS_ALPHA<0:
                THIS_RING.optimize_vector(.0001,THIS_PARA);
                RUNNING_DATA[THIS_RING.WORD_ID] = THIS_RING.VECTOR; 
            THIS_ALPHA = THIS_RING.polynomial_func(THIS_PARA); # 更新ALPHA;
            self.RING_PARA_PAIR_CC.append( [ THIS_RING,THIS_PARA, THIS_ALPHA,ALPHA  ] ); # ALPHA是期望调整值 -5代表期望小于5 5代表期望大于5;    
            RUNNING_DATA[THIS_RING.WORD_ID] = THIS_RING.VECTOR;  # 更新全局变量;
    
    def optimize_self(self):
        """ Firstly,adjust the f(x) into > alpha_0; """
        for i in range(len(self.RING_PARA_PAIR_CD)):
            self.adjust_to_excepted_value(self.RING_PARA_PAIR_CD[i]);
            self.RING_PARA_PAIR_CD[i][0].VECTOR = RUNNING_DATA[self.RING_PARA_PAIR_CD[i][0].WORD_ID]; # 更新内部VECTOR;
            self.RING_PARA_PAIR_CD[i][2] = self.RING_PARA_PAIR_CD[i][0].polynomial_func(self.RING_PARA_PAIR_CD[i][1]); # 更新内部ALPHA;
        for i in range(len(self.RING_PARA_PAIR_CC)):
            self.adjust_to_excepted_value(self.RING_PARA_PAIR_CC[i]);    
            self.RING_PARA_PAIR_CC[i][0].VECTOR = RUNNING_DATA[self.RING_PARA_PAIR_CC[i][0].WORD_ID]; # 更新内部VECTOR;
            self.RING_PARA_PAIR_CC[i][2] = self.RING_PARA_PAIR_CC[i][0].polynomial_func(self.RING_PARA_PAIR_CC[i][1]); # 更新内部ALPHA;
        """ Secondly,adjust the alpha_0 < p_min(y_n) < q_min(x_n)"""    
        P_MIN = 99999999.0; Q_MIN = 99999999.0;P_MAX = 0.0;Q_MAX = 0.0;
        CD_ID = 0;CD_MIN=0;
        # 找到结论中谓词的最小多项式值
        for RING_PARA_PAIR in self.RING_PARA_PAIR_CC:
            if Q_MIN>RING_PARA_PAIR[2]: Q_MIN=RING_PARA_PAIR[2];
            # if Q_MAX<RING_PARA_PAIR[2]: Q_MAX=RING_PARA_PAIR[2];
        # 不满足小于结论最小值的那些环的TARGET_ALPHA更新;        
        for RING_PARA_PAIR in self.RING_PARA_PAIR_CD:
            if P_MIN>RING_PARA_PAIR[2]: P_MIN=RING_PARA_PAIR[2];CD_MIN=CD_ID;# self.RING_PARA_PAIR_CD[CD_ID][3]=-Q_MIN;
            # if P_MAX<RING_PARA_PAIR[2]: P_MAX=RING_PARA_PAIR[2];self.RING_PARA_PAIR_CD[CD_ID][3]=-Q_MAX;
            CD_ID+=1;
        self.RING_PARA_PAIR_CD[CD_MIN][3]=-Q_MIN;    
        # 优化不满足小于结论最小值的那些环    
        for i in range(len(self.RING_PARA_PAIR_CD)):    
            if self.RING_PARA_PAIR_CD[i][3]!=ALPHA:
                if P_MIN>Q_MIN:
                    self.adjust_to_excepted_value(self.RING_PARA_PAIR_CD[i]);        
                    self.RING_PARA_PAIR_CD[i][0].VECTOR = RUNNING_DATA[self.RING_PARA_PAIR_CD[i][0].WORD_ID]; # 更新内部VECTOR;
                    self.RING_PARA_PAIR_CD[i][2] = self.RING_PARA_PAIR_CD[i][0].polynomial_func(self.RING_PARA_PAIR_CD[i][1]); # 更新内部ALPHA;

    def adjust_to_excepted_value(self,RING_PARA_PAIR):
        WORD_ID = RING_PARA_PAIR[0].WORD_ID;
        THIS_ALPHA = RING_PARA_PAIR[2]; # .polynomial_func(RING_PARA_PAIR[1]);
        TARGET_ALPHA = RING_PARA_PAIR[3];
        THIS_PARA = RING_PARA_PAIR[1];
        # ===========  if p(x_1,x_2)  ============
        if type(THIS_PARA)==list:
            self.adjust_to_excepted_value_2paras(RING_PARA_PAIR);
            return;
        # adjust less:
        if TARGET_ALPHA<0:
        	# 如果已经满足....
            if THIS_ALPHA<abs(TARGET_ALPHA) and TARGET_ALPHA<0:return;
            TARGET_ALPHA = (abs(TARGET_ALPHA) - ALPHA)*0.70 + ALPHA;
            if abs(THIS_ALPHA-19)<1:print('EXCEPT:  '+str(TARGET_ALPHA));
            for i in range(RUNNING_DATA[WORD_ID].size ):
                if abs(THIS_PARA[i] - RUNNING_DATA[WORD_ID][i])>1:
                    # Upadate !!!
                    RUNNING_DATA[WORD_ID][i] = THIS_PARA[i] - TARGET_ALPHA*(THIS_PARA[i] - RUNNING_DATA[WORD_ID][i])/THIS_ALPHA;
                    break;        
            return;        
        # adjust more:
        # 如果已经满足....
        if THIS_ALPHA>abs(TARGET_ALPHA)  and TARGET_ALPHA>0:return;
        TARGET_ALPHA = abs(TARGET_ALPHA)*1.40;
        # print('EXCEPT'+str(TARGET_ALPHA)+'   '+str(THIS_ALPHA));
        for i in range(RUNNING_DATA[WORD_ID].size ):
            if abs(THIS_PARA[i] - RUNNING_DATA[WORD_ID][i])>1:
                # Upadate !!!
                RUNNING_DATA[WORD_ID][i] = THIS_PARA[i] - TARGET_ALPHA*(THIS_PARA[i] - RUNNING_DATA[WORD_ID][i])/THIS_ALPHA;
                break;        
        return;            

    def adjust_to_excepted_value_2paras(self,RING_PARA_PAIR):
        WORD_ID = RING_PARA_PAIR[0].WORD_ID;
        THIS_ALPHA = RING_PARA_PAIR[2]; # .polynomial_func(RING_PARA_PAIR[1]);
        TARGET_ALPHA = RING_PARA_PAIR[3];
        THIS_PARA = RING_PARA_PAIR[1];
        # adjust less:
        # 如果已经满足....
        if THIS_ALPHA<abs(TARGET_ALPHA) and TARGET_ALPHA<0:return;
        if TARGET_ALPHA<0:
            TARGET_ALPHA = abs(TARGET_ALPHA)*0.70;
            for i in range(RUNNING_DATA[WORD_ID].size  ):
                if abs((THIS_PARA[i][0] - RUNNING_DATA[WORD_ID][i])/(THIS_PARA[i][1] - RUNNING_DATA[WORD_ID][i])    )>1:
                    # Upadate !!!
                    DELTA_1 = THIS_ALPHA*(THIS_PARA[1][i] - RUNNING_DATA[WORD_ID][i]);
                    DELTA_2 = THIS_PARA[0][i] - RUNNING_DATA[WORD_ID][i];
                    RUNNING_DATA[WORD_ID][i] = ( DELTA_2*TARGET_ALPHA*THIS_PARA[1][i] - DELTA_1*THIS_PARA[0][i])/(DELTA_2*TARGET_ALPHA - DELTA_1);
                    break;        
            return;        
        # adjust more:
        # 如果已经满足....
        if THIS_ALPHA>abs(TARGET_ALPHA) and TARGET_ALPHA>0:return;
        TARGET_ALPHA = abs(TARGET_ALPHA)*1.40;
        for i in range(RUNNING_DATA[WORD_ID].size  ):
            if abs((THIS_PARA[0][i] - RUNNING_DATA[WORD_ID][i])/(THIS_PARA[1][i] - RUNNING_DATA[WORD_ID][i])    )>1:
                # Upadate !!!
                DELTA_1 = THIS_ALPHA*(THIS_PARA[1][i] - RUNNING_DATA[WORD_ID][i]);
                DELTA_2 = THIS_PARA[0][i] - RUNNING_DATA[WORD_ID][i];
                RUNNING_DATA[WORD_ID][i] = ( DELTA_2*TARGET_ALPHA*THIS_PARA[1][i] - DELTA_1*THIS_PARA[0][i])/(DELTA_2*TARGET_ALPHA - DELTA_1);
                break;        
        return;       

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
        """ TARGET = -0.00001:  f(x) >0但希望调整至f(x)<0;
            TARGET =  0.00001:  f(x) <0但希望调整至f(x)>0;
        """
        if len(PARAMETERS)==2:
            self.optimize_vector_two(TARGET,PARAMETERS);
            return;
        if self.polynomial_func(PARAMETERS) < TARGET and TARGET>0: # TARGET >0 and f(x) <0;
            # print('adj:'+str(PARAMETERS -self.VECTOR))
            for i in range(len(self.VECTOR)):
                # print(PARAMETERS[i]-self.VECTOR[i])
                if PARAMETERS[i]-self.VECTOR[i] <0:
                    self.VECTOR[i] = PARAMETERS[i] - 1.0;
                    return;
        if self.polynomial_func(PARAMETERS) > TARGET and TARGET<0: # TARGET <0 and f(x) >0;            
            # print('adj:'+str(PARAMETERS -self.VECTOR))
            for i in range(len(self.VECTOR)):
                # print(self.VECTOR)
                if PARAMETERS[i]-self.VECTOR[i] >0:
                    self.VECTOR[i] = PARAMETERS[i] + 1.0;
                    return;

    def optimize_vector_two(self,TARGET,PARAMETERS):
        if self.polynomial_func(PARAMETERS) < TARGET and TARGET>0: # TARGET >0 and f(x,y) <0但希望调整至f(x,y)>0;
            # print(self.VECTOR)
            for i in range(len(self.VECTOR)):
                if (PARAMETERS[0][i]-self.VECTOR[i])/(PARAMETERS[1][i]-self.VECTOR[i]) <0:
                    if PARAMETERS[0][i]<self.VECTOR[i]:
                        self.VECTOR[i] = PARAMETERS[0][i] - 1.0;
                        return;
                    self.VECTOR[i] = PARAMETERS[1][i] - 1.0;
                    return;
        if self.polynomial_func(PARAMETERS) > TARGET and TARGET<0: # TARGET <0 and f(x,y) >0但希望调整至f(x,y)<0;                
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

# ------------------- BI-Relationship JUDGMENT FUNCTIONS ---------------------------

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

# -------------------- BI-Relationship OPTIMIZATION FUNCTIONS ---------------------------
"""
当不满足判定的命题条件时
如何优化调整至满足
其中命题推理样例以字典对的形式封装
"""

def print_pair(DICT_PAIR):
    DICT_LEFT  = {}; # DICT_PAIR[0];
    DICT_RIGHT = {}; # DICT_PAIR[1];
    for FUNC,VARS in DICT_PAIR[0].items():
        if type(VARS) == list:
            VARS_LIST = [];
            for VAR in VARS:VARS_LIST.append(make_data.resolve_ID(VAR,'SYMBOL'));
            DICT_LEFT[make_data.resolve_ID(FUNC,'SYMBOL')] = VARS_LIST;
            continue;
        DICT_LEFT[make_data.resolve_ID(FUNC,'SYMBOL')] = make_data.resolve_ID(VARS,'SYMBOL');                        
    for FUNC,VARS in DICT_PAIR[1].items():
        if type(VARS) == list:
            VARS_LIST = [];
            for VAR in VARS:VARS_LIST.append(make_data.resolve_ID(VAR,'SYMBOL'));
            DICT_RIGHT[make_data.resolve_ID(FUNC,'SYMBOL')] = VARS_LIST;
            continue;
        DICT_RIGHT[make_data.resolve_ID(FUNC,'SYMBOL')] = make_data.resolve_ID(VARS,'SYMBOL');    
    # print(  ( DICT_LEFT , DICT_RIGHT ) );
    LOGIC_PROP = '';
    FIRST_FLAG = 0;
    for FUNC,VARS in DICT_LEFT.items():
        if FIRST_FLAG>0:LOGIC_PROP=LOGIC_PROP+'^';
        FIRST_FLAG+=1;
        if type(VARS) == list:LOGIC_PROP=LOGIC_PROP+FUNC+'('+ VARS[0] +','+ VARS[1] +')';continue;
        LOGIC_PROP=LOGIC_PROP+FUNC+'('+ VARS +')';
    LOGIC_PROP = LOGIC_PROP + '--->';
    FIRST_FLAG = 0;
    for FUNC,VARS in DICT_RIGHT.items():
        if FIRST_FLAG>0:LOGIC_PROP=LOGIC_PROP+'^';
        FIRST_FLAG+=1;
        if type(VARS) == list:LOGIC_PROP=LOGIC_PROP+FUNC+'('+ VARS[0] +','+ VARS[1] +')';continue;
        LOGIC_PROP=LOGIC_PROP+FUNC+'('+ VARS +')';
    print(LOGIC_PROP);    
        

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
    WORD_ID = 'ADJ___132';
    WORD    = make_data.resolve_ID(WORD_ID,'SYMBOL');
    VECTOR  = make_data.resolve_ID(WORD_ID,'VECTOR');
    RING    = Ring(WORD_ID,WORD,VECTOR);
    print(RING.polynomial_func([X1,X2]));
    RING.optimize_vector(-0.00001,[X1,X2]);
    print(RING.polynomial_func([X1,X2]));

    print(RING.polynomial_func(X1));
    RING.optimize_vector(0.00001,X1);
    print(RING.polynomial_func(X1));

def TEST_optimize_logic_prop():
    """ ({'ADJ___144':'NN___17','ADV___100':'ADJ___144','PR___44':'NN___17'},{'ADJ___121':''NN___152'','ADV___100':['NN___124','NN___152']}) """
    DICT_PAIR = ({'ADJ___144':'NN___17','ADV___100':'ADJ___144','PR___44':'NN___17'},{'ADJ___121':'NN___152','ADV___101':['NN___174','NN___152']});
    print_pair(DICT_PAIR);
    read_data();
    LOGIC_PROPROGRATION = Logic_Proposition(DICT_PAIR);
    print('===== establish Ring-based Logic Proposition ======');
    print(LOGIC_PROPROGRATION.RING_PARA_PAIR_CD);
    print(LOGIC_PROPROGRATION.RING_PARA_PAIR_CC);
    print('===== execute Ring-based Logic Proposition OPTIMIZATION ======');
    LOGIC_PROPROGRATION.optimize_self();
    # LOGIC_PROPROGRATION.compute_predicate_values();
    print(LOGIC_PROPROGRATION.RING_PARA_PAIR_CD);
    print(LOGIC_PROPROGRATION.RING_PARA_PAIR_CC);

if __name__ == '__main__':
    # TEST_make_a_ring(); 
    # TEST_ring_polynomial();           
    # TEST_ring_polynomial();
    # TEST_tree_constructing(2.0,[],4,0);
    # TEST_build_bitree();
    # TEST_space_constructing_by_queue(2.0,4);
    # TEST_optimize();
    TEST_optimize_logic_prop();