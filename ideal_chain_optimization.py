#!/usr/bin/env python
# -*- coding: utf-8 -*-


__ENV__  =  'python3';
__author =  'hanss401';

import numpy as np;
import pickle;
import make_data;
import sys;
import polynomial_ring;

# ============ PUBLIC DATA ==============

RUNNING_DATA = {};
ALPHA        = 10.00;


class Proposition(object):
    """docstring for Proposition"""
    def __init__(self, RING_SET):
        super(Proposition, self).__init__()
        self.RING_SET = RING_SET; # list of rings;
        self.INFER_PROP_SET = []; # H_1 ^...^ H_n ---> C_1 ^...^ C_m;
        self.NATURELY_INFER_PROP_SET = []; # H_1 ^...^ H_n ---> H_1 ^...^ H_n-1 ---> ... ---> H_1;
        self.MIN_VALUE = 99999.0;
        self.VALUES_SET = [RING.polynomial_func(RING.PARAMETERS) for RING in self.RING_SET];

    def make_naturely_infer_set(self):
        pass;    

    def can_infer(self,PROPOSITION):
        for PREDICATE in PROPOSITION:
            if PREDICATE not in self.RING_SET:
                return False;
        return True;            

    def can_infer_by_mapping(self,PROPOSITION):
        if min(self.VALUES_SET) < min(PROPOSITION.VALUES_SET):
            return True;
        return False;    


class Ideal(object):
    """docstring for Ideal"""
    def __init__(self, arg):
        super(Ideal, self).__init__()
        self.arg = arg
        self.PROPOSITIONS_SET = [];
        self.MIN_VALUES_SET = [PROPOSITION.MIN_VALUE for PROPOSITION in self.PROPOSITIONS_SET];

    def is_in_this_ideal(self,PROPOSITION):
        if PROPOSITION.MIN_VALUE>min(self.MIN_VALUES_SET) and PROPOSITION.MIN_VALUE<max(self.MIN_VALUES_SET):
            return True;
        return False;    

    # ++++++++++   Core 🐼🐼🐼 Function   ++++++++++++++
    def optimize_this_ideal(self):
        pass;


def train_process():
    # first: read data;
    SAVED_FILE = open('expirience_set.dst','rb');
    EXPIRIENCE_SET = pickle.load(SAVED_FILE);
    LOGIC_PROP_SET = [];
    for DICT_PAIR in EXPIRIENCE_SET:
        LOGIC_PROP_SET.append(polynomial_ring.Logic_Proposition( DICT_PAIR )  );
    for LOGIC_PROP in LOGIC_PROP_SET:
        polynomial_ring.print_pair(LOGIC_PROP.DICT_PAIR);


# -------------------- MADE FUNCTIONS ----------------------



if __name__ == '__main__':
    polynomial_ring.read_data();
    train_process();            