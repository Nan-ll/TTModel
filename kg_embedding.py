import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import time
import codecs as cs

def init_nunif(sz, bnd=None):
    if bnd is None: 
        if len(sz) >= 2:
            bnd = np.sqrt(6) / np.sqrt(sz[0] + sz[1])
        else:
            bnd = 1.0 / np.sqrt(sz[0])
    return np.random.uniform(low=-bnd, high=bnd, size=sz)

class BaseModel(object):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        self.samplef = samplef 
        self.ne = ne 
        self.nr = nr 
        self.dim = dim 
        self.pairwise = kwargs.pop("pairwise", True)  
        self.epochs = kwargs.pop("epochs",200)
        self.batch_size = kwargs.pop("batch_size",1024)
        self.learning_rate = kwargs.pop("learning_rate",0.01)
        self.reg = kwargs.pop("reg",0.0)
        self.margin = kwargs.pop("margin",1.0)
        self.typr_shape = 50
        self.param_names = []
        self.E_shape = [self.ne, self.dim]
        self.R_shape = [self.nr, self.dim]
        self.TE_shape = [self.ne, self.typr_shape]
        self.TR_shape = [self.nr, self.typr_shape]
        self.reg_loss = tf.constant(0, dtype=tf.float32)

    def _on_epoch_begin(self):
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def rnegative_sample(self, fact, nr):
        my_randint = lambda n: int(n * random.random())
        return my_randint(nr)

    def _get_batch(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        pos = self.X[indices]
        neg = np.array([self.samplef(fact, self.ne) for fact in pos])
        rneg = np.array([self.rnegative_sample(fact, self.nr) for fact in pos])
        subjs = pos[:,0]
        objs = pos[:,2]
        preds = pos[:,1]
        neg_subjs = neg[:,0]
        neg_objs = neg[:,2]

        return {self.ps:subjs, self.po:objs, self.ns:neg_subjs, self.no:neg_objs, self.p:preds, self.np:rneg}

    def _add_param(self, name, shape, bnd=None):
        init_vals = init_nunif(shape, bnd)
        var = tf.Variable(init_vals, dtype=tf.float32, name=name) 
        setattr(self, name, var)
        self.param_names.append(name)
        self._regularize(var)


    def create_params(self):
        self._add_param("E", self.E_shape)
        self._add_param("R", self.R_shape)


    def gather(self, s, p, o):
        E_s = tf.gather(self.E, s)
        R = tf.gather(self.R, p)
        E_o = tf.gather(self.E, o)
        return E_s, R, E_o

    def rgather(self, p):
        R = tf.gather(self.R, p)
        return R

    def gather_np(self, si, pi, oi):
        es = self.E[si]
        eo = self.E[oi]
        r = self.R[pi]
        return es, r, eo 


    def train_score(self, s, p, o):
        raise NotImplementedError("train_score should be defined by the inheriting class")

    def ptrain_score(self, s, p, o):
        raise NotImplementedError("train_score should be defined by the inheriting class")

    def strain_score(self, s, p, o):
        raise NotImplementedError("train_score should be defined by the inheriting class")

    def _regularize(self, var):
        if self.reg > 0: 
            self.reg_loss += tf.nn.l2_loss(var)

    def train_loss(self, score_pos, score_neg):
        if self.pairwise:
            rank_loss = tf.reduce_sum(tf.maximum(0.0, self.margin - score_pos + score_neg))
        else:
            rank_loss = tf.reduce_sum(tf.nn.softplus(-score_pos) + tf.nn.softplus(score_neg))
        return rank_loss + self.reg * self.reg_loss

    def fit(self, X):

        self.X = np.array(X)
        self.ps = tf.placeholder(tf.int32, [self.batch_size])
        self.p = tf.placeholder(tf.int32, [self.batch_size])
        self.po = tf.placeholder(tf.int32, [self.batch_size])
        self.ns = tf.placeholder(tf.int32, [self.batch_size])
        self.no = tf.placeholder(tf.int32, [self.batch_size])
        self.np = tf.placeholder(tf.int32, [self.batch_size])


        self.create_params()

        score_pos = self.train_score(self.ps, self.p, self.po)
        score_neg = self.train_score(self.ns, self.p, self.no)
        self.loss = self.train_loss(score_pos, score_neg)

        pscore_pos = self.ptrain_score(self.ps, self.p, self.po)
        pscore_neg = self.ptrain_score(self.ns, self.p, self.no)

        tscore_pos = self.strain_score(self.ps, self.p, self.po)
        tscore_neg = self.strain_score(self.ps, self.np, self.po)
        self.tloss = self.train_loss(tscore_pos, tscore_neg)

        self.ploss = self.train_loss(pscore_pos, pscore_neg)
        
        self._optimize() 


    def _run_epoch(self, optimizer):
        start = time.time() 
        self._on_epoch_begin()
        tot_loss = 0
        self.cur_epoch += 1

        for b in range(len(self.X)//self.batch_size):
            feed_dict = self._get_batch(b)
            _,l = self.sess.run([optimizer,self.loss], feed_dict=feed_dict)
            tot_loss += l

        avg_loss = tot_loss / (len(self.X)//self.batch_size * self.batch_size)

        t = time.time() - start 
        print("Epoch: %i/%i; loss = %.9f (%.1f s)" %(self.cur_epoch+1,self.epochs,avg_loss,t))
        if (self.cur_epoch+1)%10 == 0:
            print("")


class BaseFeatureSumModel(BaseModel):
    def __init__(self, ne, nr, dim, samplef, W_text, W_rtext,head,tail,**kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)
        self.orig_samplef = samplef 
        self.train_words = kwargs.pop("train_words",True)
        self.reg = 0.0
        self.W_text = W_text
        self.head = head
        self.tail = tail
        self.text_dim = next(len(x) for x in W_rtext if x is not None)
        self.R_shape = [self.nr, self.text_dim + self.dim]
        self.W_rtext = W_rtext
        self.rtext_dim = next(len(x) for x in W_rtext if x is not None)


    def create_params(self):
        self.num_unknown = sum(1 for x in self.W_text if x is None)
        self.rnum_unknown = sum(1 for x in self.W_rtext if x is None)
        self.W_rtext = np.array([x for x in self.W_rtext if x is not None])
        self.W_text = np.array([x for x in self.W_text if x is not None])
        self.type_embedding = tf.Variable(init_nunif([self.ne,self.typr_shape]), dtype=tf.float32,name="type")
        self.relation_type = tf.Variable(init_nunif([self.nr, self.typr_shape]), dtype=tf.float32,name="rtype")
        self.ER = tf.Variable(init_nunif([self.ne, self.typr_shape]), dtype=tf.float32, name='ER')

        self._add_param("E", self.E_shape)
        self._add_param("R", self.R_shape)
        self._add_param("type",self.TE_shape)
        self._add_param("rtype",self.TR_shape)
        

        if self.num_unknown > 0:
            print("%i entities with unknown text embeddings" %(self.num_unknown))
            W_known = tf.Variable(self.W_text, dtype=tf.float32, trainable=self.train_words)
            W_unknown = tf.Variable(init_nunif([self.num_unknown, self.text_dim]), dtype=tf.float32)
            self.W = tf.concat([W_known,W_unknown], axis=0, name="W")
            self.phase2_vars = [W_unknown]
            if self.train_words:
                self.phase2_vars.append(W_known)
        else:
            self.W = tf.Variable(self.W_text, dtype=tf.float32, trainable=self.train_words, name="W")
            self.phase2_vars = [self.W]

        if self.rnum_unknown > 0:
            print("%i entities with unknown text embeddings" %(self.rnum_unknown))
            rW_known = tf.Variable(self.W_rtext, dtype=tf.float32, trainable=self.train_words)
            rW_unknown = tf.Variable(init_nunif([self.rnum_unknown, self.rtext_dim]), dtype=tf.float32)  #text_dim文本词向量的维度
            self.WR = tf.concat([rW_known,rW_unknown], axis=0, name="WR")
            self.phase2_vars.append(rW_unknown)
            if self.train_words:
                self.phase2_vars.append(rW_known)
        else:
            self.WR = tf.Variable(self.W_rtext, dtype=tf.float32, trainable=self.train_words, name="WR")
            self.phase2_vars.append(self.WR)

        self.param_names.append("W")
        self._regularize(self.W)

        self.param_names.append("WR")
        self._regularize(self.WR)

        self.param_names.append("ER")
        self._regularize(self.ER)

        self.M = tf.Variable(np.zeros([self.text_dim,self.dim]), dtype=tf.float32, name="M")
        self.phase2_vars.append(self.M)
        self.param_names.append("M")

        self.N = tf.Variable(np.zeros([self.rtext_dim, self.dim]), dtype=tf.float32, name="N")
        self.phase2_vars.append(self.N)
        self.param_names.append("N")
        self.W = tf.matmul(self.W, self.M)
        self.WR = tf.matmul(self.WR, self.N)

    def _optimize(self):
        self.floss = self.loss + 0.1*self.tloss + 0.3*self.ploss
        phase1_vars = tf.trainable_variables()
        if hasattr(self,"phase2_vars"):
            for var in self.phase2_vars:
                phase1_vars.remove(var)
        opt1 = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.floss, var_list=phase1_vars)
        opt2 = tf.train.AdagradOptimizer(0.01).minimize(self.floss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.cur_epoch = 0

        print("Optimizing - phase 1")
        for epoch in range(self.epochs//2):
            self._run_epoch(opt1)
        print("")

        print("Optimizing - phase 2")
        for epoch in range(self.epochs//2):
            self._run_epoch(opt2)
        print("")

        tf_objects = [getattr(self, attr) for attr in self.param_names]
        vals = self.sess.run(tf_objects)
        for attr,val in zip(self.param_names,vals):
            setattr(self, attr, val)

    def gather(self, s, p, o):
        E_s, R, E_o = BaseModel.gather(self, s, p, o)
        W_s = tf.gather(self.W, s)
        W_o = tf.gather(self.W, o)
        W_p = tf.gather(self.WR, self.p)
        ht = tf.gather(self.ER,s)*tf.gather(self.type,s)
        tt = tf.gather(self.ER,o)*tf.gather(self.type,s)
        rt = tf.gather(self.relation_type,p)
        return E_s, R, E_o,W_s,W_o,ht,tt,rt,W_p

class TransEFeatureSum(BaseFeatureSumModel):
    def __init__(self, ne, nr, dim, samplef,W_text, W_rtext,head,tail, **kwargs):
        BaseFeatureSumModel.__init__(self, ne, nr, dim, samplef, W_text,W_rtext,head,tail, **kwargs)
        self.R_shape = [self.nr, self.dim]
        self.train_words = True

    def ptrain_score(self,s, p, o):
        E_s, R, E_o,W_s,W_o,ht,tt,rt,W_p = self.gather(s, p, o)
        return -tf.reduce_sum(tf.abs(ht + rt - tt), axis=-1)

    def get_pair(self,p):
        head = tf.convert_to_tensor(np.array([np.random.choice(i, size=16) for i in self.head]))
        tail = tf.convert_to_tensor(np.array([np.random.choice(i, size=16) for i in self.tail]))
        head_pair = tf.gather(head,p)
        tail_pair = tf.gather(tail,p)
        head_pair = tf.gather(self.type_embedding,head_pair)
        tail_pair = tf.gather(self.type_embedding,tail_pair)
        return head_pair,tail_pair

    def euclidean_distance_by_tf(self,vector1, vector2):
        temp_vector1 = vector1 - vector2
        temp_vector2 = tf.square(temp_vector1)
        s = tf.reduce_sum(temp_vector2)
        return tf.sqrt(s)


    def strain_score(self,s,p,o):
        E_s, R, E_o, W_s, W_o, ht, tt, rt,W_p = self.gather(s, p, o)
        head_pair,tail_pair = self.get_pair(p)
        ht = tf.expand_dims(ht,-2)
        tt = tf.expand_dims(tt,-2)
        fs1 = tf.abs(ht-head_pair)
        fs2 = tf.abs(tt-tail_pair)
        return -tf.reduce_sum((fs1+fs2)/2,axis=-1)

    def train_score(self, s, p, o):
        E_s, R, E_o,W_s,W_o,ht,tt,rt,W_p = self.gather(s, p, o)
        E_s = E_s + W_s
        E_o = E_o + W_o
        # R = R + W_p
        return -tf.reduce_sum(tf.abs(E_s + R - E_o), axis=-1)

    def score(self, si, pi, oi):
        es = self.E[si] + self.W[si]
        eo = self.E[oi] + self.W[oi]
        r = self.R[pi]
        fe = -np.sum(np.abs(es + r - eo), axis=-1)
        ftype = -np.sum(np.abs(self.type[si] + self.rtype[pi] - self.type[oi]), axis=-1)
        return fe + 0.1*ftype


