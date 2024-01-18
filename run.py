# -*- coding: utf-8 -*-
import pickle 
import os 
import random 
import argparse
import ctypes
import numpy as np
import nltk 
import kg_embedding
import codecs as cs
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
from tqdm import tqdm

my_randint = lambda n : int(n * random.random())

def negative_sample(fact, ne):
    if random.random() < 0.5:
        return (my_randint(ne), fact[1], fact[2])
    else:
        return (fact[0], fact[1], my_randint(ne))

class RunKGText(object):
    def __init__(self):
        self._load_text()
        self._load_graph()

        print("Loaded:")
        print("\t%i entities" %(len(self.entities)))
        print("\t%i relations" %(len(self.relations)))
        num_train = len([1 for s,p,o,d in self.facts if d & (1<<0)])
        num_valid = len([1 for s,p,o,d in self.facts if d & (1<<1)])
        num_test = len([1 for s,p,o,d in self.facts if d & (1<<2)])
        print("\t%i training facts" %(num_train))
        print("\t%i validation facts" %(num_valid))
        print("\t%i testing facts" %(num_test))

        path = './data/DTI/relation_des.txt'
        self._load_relation(path)
        self.W_rtext = self._build_relation_data()
        self.W_text = self._build_text_data()

    def _load_relation(self, path):
        print("Loading text data")
        self.relation_vocab = set([])
        self.relation2name = {}
        with open(path, "r",encoding='utf-8') as f:
            file = f.read()
            sll = file.split('\n')[:-1]
            for line in sll:
                self.relation2name[line] = line.split("\t")[0]
                self.relation_vocab = self.relation_vocab.union(set(line.split("\t")[0]))

    def _build_relation_data(self):
        W_rtext = []
        for e in self.relations:
            e = e.replace("_"," ")
            if e != 'DRUG TARGET':
               vectors = np.array(self.sentence_vec[e])
            else:
                vectors = np.array(self.sentence_vec[e])
            if len(vectors) == 0:
                W_rtext.append(None)
            else:
                W_rtext.append(vectors)
        return W_rtext

    def _load_graph(self):
        print("Loading knowledge graph") 
        self.facts = []
        self.triple = []
        self.entities = set([])
        self.relations = set([])
        self.relation2id = {}
        self.entity2id = {}
        self.test = []

        for i,fname in enumerate(["train.txt","valid.txt","test.txt"]):
            fname = os.path.join("data","DTI",fname)
            with open(fname) as f:
                file = f.read().split('\n')[:-1]
                for line in file:
                    subj, pred, obj = line.split('\t')
                    self.facts.append((subj, pred, obj, 1<<i))
                    self.triple.append((subj, pred, obj))
                    if pred == 'DRUG_TARGET':
                        self.test.append([subj,pred,obj])
                    self.entities.add(subj)
                    self.entities.add(obj)
                    self.relations.add(pred)
        self.entities = list(self.entities)
        self.relations = list(self.relations)

        self.entity2id = dict(zip(self.entities, range(len(self.entities))))
        self.relation2id = dict(zip(self.relations, range(len(self.relations))))

        self.facts = [(self.entity2id[s], self.relation2id[p], self.entity2id[o], d) for s, p, o, d in
                      self.facts]
        self.triple = [(self.entity2id[s], self.relation2id[p], self.entity2id[o]) for s, p, o in
                      self.triple]
        self.head = []
        self.tail = []
        for i in range(len(self.relations)):
            mhead = []
            mtail = []
            for h,r,t in self.triple:
                if r==i and h not in mhead:
                    mhead.append(h)
                if r==i and t not in mtail:
                    mtail.append(t)
            self.head.append(mhead)
            self.tail.append(mtail)

        self.test = np.array([(s, p, o) for s, p, o in self.test])


    def _load_text(self):
        print("Loading text data")
        self.entity2word = {}
        with open("data/DTI/totalentity.pkl","rb") as f:
            self.google_vecs = pickle.load(f)

        with open("data/DTI/totalrelation.pkl","rb") as fr:
            self.sentence_vec = pickle.load(fr)

        fname = os.path.join("data","DTI","descriptions.txt")

        with open(fname,encoding='utf-8') as f:
            for line in f:
                entity, name, desc = line.split("\t")
                self.entity2word[entity] = name


    def _build_text_data(self):
        W_text = []
        for e in self.entities:
            word = self.entity2word[e]
            vectors = np.array(self.google_vecs[word])
            if len(vectors) == 0:
                W_text.append(None)
            else:
                W_text.append(vectors)

        return W_text

    def run_embedding(self, dim, **kwargs):

        print("Training")
        X_train = [(s,p,o) for s,p,o,d in self.facts if d & (1<<0)]

        cls = eval("kg_embedding."+"TransEFeatureSum")
        self.model = cls(len(self.entities), len(self.relations), dim, negative_sample, self.W_text, self.W_rtext,self.head,self.tail,**kwargs)
        self.model.fit(X_train)


    def get_ent_indices(self, ent_labels):

        if type(ent_labels) == str:
            return self.entity2id[ent_labels]
        else:
            return np.array([self.entity2id[l] for l in ent_labels])

    def ROC_evaluate(self):
        print("================================================================================")
        test_data = [(s, p, o) for s, p, o, d in self.facts if d & (1 << 2)]
        pse_list = set(list(np.array(test_data)[:, 1]))
        pse = list(np.array(test_data)[:, 1])[0]
        se_facts_full_dict = {se: set() for se in self.facts if se[1]==pse}
        pse_drugs = list(set(list(np.concatenate([self.test[:, 0], self.test[:, 2]]))))
        drug_combinations = np.array([[d1, d2] for d1, d2 in list(itertools.product(pse_drugs, pse_drugs)) if d1 != d2])
        d1 = np.array(self.get_ent_indices(list(drug_combinations[:, 0]))).reshape([-1, 1])
        d2 = np.array(self.get_ent_indices(list(drug_combinations[:, 1]))).reshape([-1, 1])
        drug_combinations = np.concatenate([d1, d2], axis=1)
        se_auc_roc_list = []
        se_auc_pr_list = []

        for se in tqdm(pse_list, desc="Evaluating test data for each side-effect"):
            se_all_facts_set = se_facts_full_dict
            se_test_facts_pos = np.array([[s, p, o] for s, p, o in test_data if p == se])
            se_test_facts_pos_size = len(se_test_facts_pos)

            se_test_facts_neg = np.array([[d1, se, d2] for d1, d2 in drug_combinations
                                          if (d1, se, d2) not in se_all_facts_set
                                          and (d2, se, d1) not in se_all_facts_set])
            se_test_facts_neg = se_test_facts_neg[:se_test_facts_pos_size, :]
            set_test_facts_all = np.concatenate([se_test_facts_pos, se_test_facts_neg])
            se_test_facts_labels = np.concatenate(
                [np.ones([len(se_test_facts_pos)]), np.zeros([len(se_test_facts_neg)])])
            subjs = list(map(int, set_test_facts_all[:, 0].tolist()))
            objs = list(map(int, set_test_facts_all[:, 2].tolist()))
            preds = list(map(int, set_test_facts_all[:, 1].tolist()))
            se_test_facts_scores = self.model.score(subjs, preds, objs)
            se_auc_pr = average_precision_score(se_test_facts_labels, se_test_facts_scores)
            se_auc_roc = roc_auc_score(se_test_facts_labels, se_test_facts_scores)

            se_auc_roc_list.append(se_auc_roc)
            se_auc_pr_list.append(se_auc_pr)
            print("AUC-ROC: %1.4f - AUC-PR: %1.4f" %
                  (se_auc_roc, se_auc_pr), flush=True)
        se_auc_roc_list_avg = np.average(se_auc_roc_list)
        se_auc_pr_list_avg = np.average(se_auc_pr_list)

        print("================================================================================")
        print("[AVERAGE]AUC-ROC: %1.4f - AUC-PR: %1.4f" %
              (se_auc_roc_list_avg, se_auc_pr_list_avg), flush=True)
        print("================================================================================")

if __name__ == "__main__":
    runner = RunKGText()
    runner.run_embedding(dim=100, epochs=100, batch_size=1024, margin=1)
    runner.ROC_evaluate()

