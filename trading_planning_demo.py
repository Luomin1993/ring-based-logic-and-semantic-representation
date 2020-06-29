#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data import *;
import sys;
sys.path.append('..');
# from polynomial_ring import *

__ENV__  =  'python3';
__author =  'hanss401';
__date__ =  '20200629';

# ========================================    Method Functions ==========================================

class Logic_Proposition(object):
    """docstring for Logic_Proposition"""
    def __init__(self, DICT_PAIR):
        super(Logic_Proposition, self).__init__()
        self.DICT_PAIR  = DICT_PAIR;
        self.CONDITIONS = [];
        self.CONCLUSIONS = [];
        self.RING_PARA_PAIR_CD = [];  # 条件...
        self.RING_PARA_PAIR_CC = [];  # 结论...
        # self.compute_predicate_values();

    def compute_predicate_values(self):
        for PREDICATE,NOUN in self.DICT_PAIR[0].items():
            THIS_WORD = ID_TO_WORD_DICT[PREDICATE];
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
            # BUG_DEBUG(PREDICATE + ' ---- ' + str(RUNNING_DATA['PR___147']));
            THIS_WORD = ID_TO_WORD_DICT[PREDICATE];
            THIS_ID   = PREDICATE;
            THIS_VEC  = RUNNING_DATA[PREDICATE];
            if type(NOUN)==list: THIS_PARA = [ RUNNING_DATA[NOUN[i]] for i in range(2)];
            else: THIS_PARA = RUNNING_DATA[NOUN];
            THIS_RING = Ring(THIS_ID,THIS_WORD,THIS_VEC);
            # 执行optimize_vector;使谓词逻辑为真:
            THIS_ALPHA = THIS_RING.polynomial_func(THIS_PARA);
            if THIS_ALPHA<0:
                THIS_RING.optimize_vector(.0001,THIS_PARA);
                RUNNING_DATA[THIS_RING.WORD_ID] = THIS_RING.VECTOR;  # 更新全局变量;
            THIS_ALPHA = THIS_RING.polynomial_func(THIS_PARA); # 更新ALPHA;
            self.RING_PARA_PAIR_CC.append( [ THIS_RING,THIS_PARA, THIS_ALPHA,ALPHA  ] ); # ALPHA是期望调整值 -5代表期望小于5 5代表期望大于5;    
            RUNNING_DATA[THIS_RING.WORD_ID] = THIS_RING.VECTOR;  # 更新全局变量;
    
    def optimize_self(self):
        """ Zeroznly,adjust the f(x) > 0; """
        self.compute_predicate_values();
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
            for i in range(RUNNING_DATA[WORD_ID].size ):
                if abs(THIS_PARA[i] - RUNNING_DATA[WORD_ID][i])>1:
                    # Upadate !!!
                    RUNNING_DATA[WORD_ID][i] = THIS_PARA[i] - TARGET_ALPHA*(THIS_PARA[i] - RUNNING_DATA[WORD_ID][i])/THIS_ALPHA;
                    break;        
            return;        
        # adjust more:
        # 如果已经满足....
        if THIS_ALPHA>abs(TARGET_ALPHA) and TARGET_ALPHA>0:return;
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
                if abs((THIS_PARA[0][i] - RUNNING_DATA[WORD_ID][i])/(THIS_PARA[1][i] - RUNNING_DATA[WORD_ID][i])    )>1:
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


class Ring(object):
    """ Ring-based Logic """
    def __init__(self, WORD_ID,WORD,VECTOR):
        super(Ring, self).__init__()
        self.WORD_ID      =    WORD_ID;
        self.VECTOR       =     VECTOR;
        self.PARAMETERS   =       None;
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


# ========================  DEMO APP ==========================

ALPHA        = 10.00;
RUNNING_DATA = {};
ID_TO_WORD_DICT = {};
WORD_TO_ID_DICT = {};
ADV_WORDS = [];
ADJ_WORDS = [];
PR_WORDS = [];
PR_WORDS_2 = ['做多','做空','看跌','看涨'];
PR_WORDS_1 = ['上涨','下跌'];
NN_WORDS_P = ['散户','庄家','机构'];
NN_WORDS = [];

def read_data_and_words():
    """Read the static data into this;"""
    ADV_MAT = np.load('ADV.npy');
    ADJ_MAT = np.load('ADJ.npy');
    PR_MAT = np.load('PR.npy');    
    NN_MAT = np.load('NN.npy');
    with open('ADV.dst','r') as FILE_STREAM:ADV_WORDS = FILE_STREAM.readlines();
    with open('ADJ.dst','r') as FILE_STREAM:ADJ_WORDS = FILE_STREAM.readlines();
    with open('PR.dst','r') as FILE_STREAM:PR_WORDS = FILE_STREAM.readlines();
    with open('NN.dst','r') as FILE_STREAM:NN_WORDS = FILE_STREAM.readlines();
    for i in range(ADV_MAT.shape[0]): RUNNING_DATA['ADV___'+str(i)] = ADV_MAT[i]; ID_TO_WORD_DICT['ADV___'+str(i)] = ADV_WORDS[i].replace('\n','') ; WORD_TO_ID_DICT[ADV_WORDS[i].replace('\n','') ] = 'ADV___'+str(i);
    for i in range(ADJ_MAT.shape[0]): RUNNING_DATA['ADJ___'+str(i)] = ADJ_MAT[i]; ID_TO_WORD_DICT['ADJ___'+str(i)] = ADJ_WORDS[i].replace('\n','') ; WORD_TO_ID_DICT[ADJ_WORDS[i].replace('\n','') ] = 'ADJ___'+str(i);
    for i in range(PR_MAT.shape[0]): RUNNING_DATA['PR___'+str(i)] = PR_MAT[i]; ID_TO_WORD_DICT['PR___'+str(i)] =  PR_WORDS[i].replace('\n','') ; WORD_TO_ID_DICT[PR_WORDS[i].replace('\n','') ] =  'PR___'+str(i);
    for i in range(NN_MAT.shape[0]): RUNNING_DATA['NN___'+str(i)] = NN_MAT[i]; ID_TO_WORD_DICT['NN___'+str(i)] =  NN_WORDS[i].replace('\n','') ; WORD_TO_ID_DICT[NN_WORDS[i].replace('\n','') ] =  'NN___'+str(i);

def training():
    # first: read data;
    LOGIC_PROP_SET = [];
    for DICT_PAIR in EXPIRIENCE_DICT_PAIR:
        LOGIC_PROP_SET.append(Logic_Proposition( words_dict_piar_2_ids_dict_pair(DICT_PAIR) )  ); 
    # second: training;
    for LOGIC_PROP in LOGIC_PROP_SET:
        # polynomial_ring.print_pair(LOGIC_PROP.DICT_PAIR);
        print('===== establish Ring-based Logic Proposition ======');
        print_expirience(LOGIC_PROP.DICT_PAIR);
        print('===== execute Ring-based Logic Proposition OPTIMIZATION ======');
        LOGIC_PROP.optimize_self();
        print(LOGIC_PROP.RING_PARA_PAIR_CD);
        print(LOGIC_PROP.RING_PARA_PAIR_CC);    

def words_dict_piar_2_ids_dict_pair(WORDS_DICT_PAIR):
    ID_DICT_PAIR = ({},{});
    for i in range(2):
        for PR_WORD,NN_WORD in WORDS_DICT_PAIR[i].items():
            if type(NN_WORD)==list:
                ID_DICT_PAIR[i][ WORD_TO_ID_DICT[PR_WORD] ] = [ WORD_TO_ID_DICT[NN_WORD_i] for NN_WORD_i in NN_WORD ];
                continue;
            ID_DICT_PAIR[i][ WORD_TO_ID_DICT[PR_WORD] ] = WORD_TO_ID_DICT[NN_WORD];    
    return ID_DICT_PAIR;        

def valuable_events_constructing(FUTURE_CLASS):
    MODE_1 = {'PR_WORD':FUTURE_CLASS, 'ADV_WORD':'PR_WORD'};
    MODE_2 = {'PR_WORD':['NN_WORD', FUTURE_CLASS]};
    VALUABLE_EVENTS = [];
    for PR_WORD1 in PR_WORDS_1:
        for ADV_WORD in ADV_WORDS:
            VALUABLE_EVENTS.append( {PR_WORD1:FUTURE_CLASS,ADV_WORD:PR_WORD1} );
    for PR_WORD2 in PR_WORDS_2:
        for NN_WORD in NN_WORDS_P:
            VALUABLE_EVENTS.append( {PR_WORD2:[NN_WORD, FUTURE_CLASS]} );
    return VALUABLE_EVENTS;        


def news_events_constructing():
    pass;

def compute_event_values(EVENT_DICT):
    EVENT_VALUES = [];
    for PR_WORD,NN_WORD in EVENT_DICT.items():
        VEC_PR = RUNNING_DATA[WORD_TO_ID_DICT[PR_WORD]];
        if type(NN_WORD)==list:VEC_NN = [RUNNING_DATA[WORD_TO_ID_DICT[NN_WORD_i]]  for NN_WORD_i in NN_WORD  ];
        else: VEC_NN = RUNNING_DATA[WORD_TO_ID_DICT[NN_WORD]];
        THIS_RING = Ring(WORD_TO_ID_DICT[PR_WORD],PR_WORD,VEC_PR);
        EVENT_VALUES.append(THIS_RING.polynomial_func(VEC_NN) );
    return EVENT_VALUES;    

def event_reasoning(CONDITIONS_DICT):
    # find out CONCLUSIONS_SET s.t. "CONDITIONS_DICT ---> CONCLUSIONS_SET";
    CONCLUSIONS_EVENTS = [];
    CONDITIONS_VALUES = compute_event_values(CONDITIONS_DICT);
    for PR_WORD,NN_WORD in CONDITIONS_DICT.items():
        if type(NN_WORD)==list:
            if 'NN___' in WORD_TO_ID_DICT[NN_WORD[1]]:
                VALUABLE_EVENTS = valuable_events_constructing(NN_WORD[1]);
                break;
            continue;    
        if 'NN___' in WORD_TO_ID_DICT[NN_WORD]:
            VALUABLE_EVENTS = valuable_events_constructing(NN_WORD);
            break;
    for CONCLUSIONS_DICT in VALUABLE_EVENTS:
        CONCLUSIONS_VALUES = compute_event_values(CONCLUSIONS_DICT);
        if max(CONDITIONS_VALUES)<min(CONCLUSIONS_VALUES):
            CONCLUSIONS_EVENTS.append(CONCLUSIONS_DICT);
    return CONCLUSIONS_EVENTS;        


def print_expirience(DICT_LIST_PAIR):
    NEWS_EVENT = DICT_LIST_PAIR[0];
    VALUABLE_EVENTS = DICT_LIST_PAIR[1];
    print('        =========================   由新闻消息预测     =====================    ');
    print('新闻消息 : ' + str(NEWS_EVENT));
    print('**************************************************')
    if len(VALUABLE_EVENTS)==0:print('预测不出什么');
    else:
        for EVENT_DICT in VALUABLE_EVENTS:
            print('预测 : ' + str(EVENT_DICT));

def main():
    # news_events_constructing();
    read_data_and_words();
    training();
    for NEWS_EVENT in NEWS_EVENTS_DICT:print_expirience(   (NEWS_EVENT , event_reasoning(NEWS_EVENT))    );
    
if __name__ == '__main__':
    main();