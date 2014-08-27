#!/usr/bin/env python
import numpy
from traces import TraceHolder
from tiles import CollisionTable, loadtiles, tiles
from _ast import Num
from numpy import sort
from verifier import Verifier


class Learner(object):
    total_loss = 0
    num_steps = 0
    prediction = 0
    theta = list()
    
    def update(self, features, target=None):
        pass
    
    def set_weights(self):
        # update the weight given weight_init
        pass
    
    def predict(self, x):
        # calculate the prediction 
        pass
    
    def reset_loss(self):
        # reset accumulated loss but keep learned weights
        self.total_loss = 0.0
        self.num_steps = 0
        try:
            self.alpha = self.alpha_init
        except:
            pass

    def reset(self):
        # reset everything
        self.reset_loss()
        self.set_weights()

    def set_alpha(self, alpha, alpha_decay = None):
        self.alpha = alpha
        if alpha_decay is None or alpha_decay < 0:
            self.alpha = alpha
            self.alpha_init = alpha
            self.alpha_decay = None
        else:
            raise Exception("Set_Alpha not INITIALIZED for alpha decay")
    
    def loss(self):
        pass
    
class Partition_Tree_Learner(Learner):   
    """
    Re-implementation of Anna's Partition Tree Learning
    
    Theoretical Properties:
        
    """

    def __init__(self, depth, learner_factory, weight_reset=True):
        self.learner_factory = learner_factory
        self.depth = depth
        self.prior = numpy.array([2 ** -min(i, self.depth) for i in range(self.depth+1, 0, -1)])
        self.log_prior = numpy.log(self.prior)
        self.weights = None
        self.weight_reset = weight_reset
        self.reset_all_nodes()
            
    def get_max_height(self):
        # Get the height of the maximum completed subtree during this step
        if self.num_steps > 0:
            return mscb(self.num_steps-1)
        else:
            return -1
    
    def predict(self, x):
    # the order matters---learners are updated in get_learner_weighting
        p = self.get_learner_predictions(x) 
        self.w = self.get_learner_weighting()
        return self.w.dot(p)
        
    def update(self, features, target):
        """
        Update the PTL model given the input
        After the update, the nodes will have up to their full number of steps
        """
        # ANNOTATE
        max_height = self.get_max_height()
        
        if max_height > self.depth:
            raise NotImplementedError("Must set depth less than: "+ str(self.depth) + ". Currently is: " + str(max_height))
        
        self.num_steps += 1
        
        # go over all the nodes and update each one
        for number_parameters in self.nodes:
            number_parameters.update(features, target) # I presume target is a psuedo  reward
        
        # total error is defined as the error at the most complete node
        self.total_loss = self.nodes[self.depth].total_loss
        
    def reset_all_nodes(self):
        """
        This does a complete reset of the nodes
        """
        self.nodes = [Partition_Tree_Learner_Node(self.learner_factory, 0, weight_reset=self.weight_reset)]
        for i in range(self.depth): 
            self.nodes.append(Partition_Tree_Learner_Node(self.learner_factory, i+1, 
                                      child=self.nodes[i], 
                                      weight_reset=self.weight_reset))
        self.reset_loss()
    
    def get_partial_totals(self): 
        #returns a list of the total loss for each node
        return [number_parameters.total_loss for number_parameters in self.nodes[::-1]]
    
    def get_completed_totals(self):
        #returns a list of the completed subtree nodes
        return [number_parameters.prev_loss for number_parameters in self.nodes[::-1]]
    
    def get_learner_predictions(self, x):
        #returns a list of each learners predictions          
        return [number_parameters.predict(x) for number_parameters in self.nodes]

    def get_learner_weighting(self, debug=False):
        """
        Return the normalized weights for each of the learners
        """
        if debug:
            print("Squash it!")
        wc = numpy.cumsum(self.get_completed_totals())
        wp = self.get_partial_totals()

        # back in the default order
        w = (wc+wp)[::-1]

        loss = w - self.log_prior
        norm_w = numpy.exp(-loss - log_sum_exp(-loss))
        return norm_w

    def loss(self, x, r, prev_state=None):
        """
        Returns the TD error assuming reward r given for 
        transition from prev_state to x
        If prev_state is None will use leftmost element in exp_queue
        """
        if prev_state is None:
            if len(self.exp_queue) < self.horizon:
                return None
            else:
                prev_state = self.exp_queue[0][0]

        vp = r + self.gamma * self.value(x)
        v = self.value(prev_state)
        delta = vp - v
        return delta
    
class Partition_Tree_Learner_Node(object):
    """
    Used by PTL to keep track of a specific binary partition point
    """
    
    def __init__(self, learner_factory, height, child=None, weight_reset=True):
        self.height = height
        self.max_steps = 2**height
        self.learner_factory = learner_factory
        self.total_loss = 0.0 # the total loss for this period
        self.learner = None
        self.weight_reset = weight_reset
        self.child = child
        self.num_steps = 0
        self.reset()
        
    def reset_node(self):
        # reset the learner and store the completed loss
        self.prev_loss = self.total_loss
        
        if not self.learner: # if the learner is not instantiated
            self.learner = self.learner_factory()
        else:
            self.learner.reset_loss()
            if self.weight_reset: # some instances don't need to be reset
                self.learner.set_weights()
                
        self.total_loss = self.learner.total_loss # pull the loss up from the learner
        
    def reset_loss(self):
        self.total_loss = 0.0
        self.num_steps = 0
    
    def reset(self):
        # Reset all the things
        self.reset_node()
        self.reset_loss()
        self.prev_loss = 0
    
    def calculate_loss(self):
        # ANNOTATE : HOW DOES THIS WORK?
        if not self.child:
            return self.learner.total_loss
        else:
            nosplit = -self.learner.total_loss
            split = -self.child.prev_loss - self.child.total_loss
    # what is NP and log_sum_exp
            return numpy.log(2) - log_sum_exp([nosplit, split])
    
    def update(self, features, target):
        if self.check_partition_end():
            self.reset_node()
            if self.child:
                self.child.reset_completed()
        
        self.learner.update(features, target)
        self.total_loss = self.calculate_loss()
        self.num_steps = self.learner.num_steps
        
    def reset_completed(self): # WHAT IS THIS FOR?
        self.prev_loss = 0.0
    
    def check_partition_end(self):
        return self.num_steps >= self.max_steps
    
    def set_weights(self, weights):
        self.learner.set_weights(weights)
    
    def predict(self, features):
        return self.learner.predict(features)  
        
class TDLambdaLearner(Learner):
    """
    Note: the TileCoder is Rich's Python version, which is still in Alpha.
    See more at: http://webdocs.cs.ualberta.ca/~sutton/tiles2.html#Python%20Versions
    
        Collision Table notes:
            cTableSize is the size that the collision table will be instantiated to. The size must be  a power of two.
            In calls for get tiles, the collision table is used in stead of memory_size, as it already has it.
    
    """
    def __init__(self, numTilings = 1, parameters = 2, rlAlpha = 0.5, rlLambda = 0.9, rlGamma = 0.9, cTableSize=0):
        """ If you want to run an example of the code, simply just leave the parameters blank and it'll automatically set based on the parameters. """
        self.numTilings = numTilings
        self.tileWidths = list()
        self.parameters = parameters
        self.rlAlpha = rlAlpha
        self.rlLambda = rlLambda
        self.rlGamma = rlGamma
    
        self.prediction = None
        self.lastS = None
        self.lastQ = None
        self.lastPrediction = None
        self.lastReward = None
        self.traceH = TraceHolder((self.numTilings**(self.parameters)+1), self.rlLambda, 1000)
        self.F = [0 for item in range(self.numTilings)] # the indices of the returned tiles will go in here
        self.theta = [0 for item in range((self.numTilings**(self.parameters+1))+1)] # weight vector.
        self.cTable = CollisionTable(cTableSize, 'safe') # look into this...
        self.verifier = Verifier(self.rlGamma)
        
    def update(self, features, target=None):
        if features != None:
            self.learn(features, target)
            return self.prediction
        else: return None
     
    def learn(self, state, reward):
        self.loadFeatures(state, self.F)
        currentq = self.computeQ()
        if self.lastS != None:
            delta = reward - self.lastQ
            delta += self.rlGamma * currentq
            amt = delta * (self.rlAlpha / self.numTilings)
            for i in self.traceH.getTraceIndices():
                self.theta[i] += amt * self.traceH.getTrace(i)
            self.traceH.decayTraces(self.rlGamma)
            self.traceH.replaceTraces(self.F)
        self.lastQ = currentq
        self.lastS = state
        self.prediction = currentq
        self.num_steps+=1
        self.verifier.updateReward(reward)
        self.verifier.updatePrediction(self.prediction)
        

    def computeQ(self):
        q = 0
        for i in self.F:
            q += self.theta[i]
        return q
    
    def loadFeatures(self, stateVars, featureVector):
        loadtiles(featureVector, 0, self.numTilings, self.numTilings**(self.parameters), stateVars)
        print "featureVector " + str(len(self.theta))
        """ 
        As provided in Rich's explanation
               tiles                   ; a provided array for the tile indices to go into
               starting-element        ; first element of "tiles" to be changed (typically 0)
               num-tilings             ; the number of tilings desired
               memory-size             ; the number of possible tile indices
               floats                  ; a list of real values making up the input vector
               ints)                   ; list of optional inputs to get different hashings
        """
    
    def loss(self, x, r, prev_state=None):
        """
        Returns the TD error assuming reward r given for 
        transition from prev_state to x
        If prev_state is None will use leftmost element in exp_queue
        """
        if prev_state is None:
            if len(self.exp_queue) < self.horizon:
                return None
            else:
                prev_state = self.exp_queue[0][0]

        vp = r + self.gamma * self.value(x)
        v = self.value(prev_state)
        delta = vp - v
        return delta 
    
    
    def predict (self,x):
        self.loadFeatures(x, self.F)
        return self.computeQ()

class True_Online_TD2(TDLambdaLearner):
    """
        True online TD implementation
            * Has Dutch traces
    """
    def __init__(self, numTilings = 2, parameters = 2, rlAlpha = 0.5, rlLambda = 0.9, rlGamma = 0.9, cTableSize=0):
        self.numTilings = numTilings
        self.tileWidths = list()
        self.parameters = parameters
        self.rlAlpha = rlAlpha
        self.rlLambda = rlLambda
        self.rlGamma = rlGamma
    
        self.prediction = None
        self.lastS = None
        self.lastQ = None
        self.lastPrediction = None
        self.lastReward = None
        self.F = [0 for item in range(self.numTilings)]
        self.F2 = [0 for item in range(self.numTilings)]
        self.theta = [0 for item in range((self.numTilings**(self.parameters+1))+1)]
        self.cTable = CollisionTable(cTableSize, 'safe')
        self.update(None, None)
        self.e = [0 for item in range((self.numTilings**(self.parameters+1))+1)]
        self.verifier = Verifier(self.rlGamma)
        
    def update(self, features, target=None):
        if features != None:
            self.learn(features, target)
            return self.prediction
        else: return None
    
    def learn(self, state, reward):
        
        self.loadFeatures(state, self.F)
        self.currentq = 0
        for i in self.F: # create V(s)
            self.currentq += self.theta[i] 
        
        if self.lastS != None:
            delta = reward + self.rlGamma * self.currentq - self.lastQ # create delta
            
            self.loadFeatures(self.lastS, self.F2) 
            lastQ_2 = 0
            for i in self.F2:
                lastQ_2 += self.theta[i] # create new 
            
            for i in range(len(self.e)):
                self.e[i] *= self.rlGamma*self.rlGamma
            ephi = 0
            for i in self.F2:
                ephi += self.e[i]
                
            for i in self.F2:
                self.e[i] += self.rlAlpha*(1-self.rlGamma*self.rlLambda*ephi)
            
            for i in self.F2:
                self.theta[i] += self.rlAlpha*(self.lastQ - lastQ_2)
                
            for i in range(len(self.theta)):
                self.theta[i] += delta*self.e[i]
                
        self.lastQ = self.currentq
        self.lastS = state
        self.prediction = self.currentq
        self.num_steps+=1
        self.verifier.updateReward(reward)  
        self.verifier.updatePrediction(self.prediction)
        
class SwitchingLearner_bento(Learner):
    
    def __init__(self):
        #traits lib look into it 
        self.tdLambda_hand = TDLambdaLearner()
        self.tdLambda_wristRotation = TDLambdaLearner()
        self.tdLambda_wristFlexion = TDLambdaLearner()
        self.tdLambda_elbow = TDLambdaLearner()
        self.tdLambda_shoulder = TDLambdaLearner()
        
    def loadFeatures(self, stateVars):
        raise Exception("NOT IMPLEMENTED -- NOT USED")
    
    def update(self, features, target):
        self.tdLambda_hand.update(features,target[0])
        self.tdLambda_wristRotation.update(features,target[1])
        self.tdLambda_wristFlexion.update(features,target[2])
        self.tdLambda_elbow.update(features,target[3])
        self.tdLambda_shoulder.update(features,target[4])
    
    def set_weights(self):
        # update the weight given weight_init
        raise Exception("NOT IMPLEMENTED -- NOT USED")
    
    def predict(self, x):
        # predict the order of joints
        jointSwitchingJoints = numpy.array([[self.tdLambda_hand.predict(x),"Hand"], [self.tdLambda_wristRotation.predict(x),"Wrist_Rotation"], [self.tdLambda_wristFlexion.predict(x), "Wrist_Flexion"], [self.tdLambda_elbow.predict(x),"Elbow"], [self.tdLambda_shoulder.predict(x), "Shoulder"]])
        jointSwitchingJoints = sort(jointSwitchingJoints, 1)
        return jointSwitchingJoints
    
    def loss(self):
        raise Exception("NOT IMPLEMENTED -- NOT USED")
                
"""
****************************************************************
                    UTILITY FUNCTIONS
****************************************************************
"""
                
def mscb(t):
    """
    Find the index of the most significant change bit,
    the bit that will change when t is incremented
    """
    return int(numpy.log2(t ^ (t + 1)))

def log_sum_exp(v):
    """
    Calculate the log of the sum of exponentials of the vector elements.

    Use the shifting trick to minimize numerical precision errors.
    Argument is a list-like thing
    """
    number_actions = max(v)
    x = number_actions * numpy.ones(numpy.size(v))
    return number_actions + numpy.log(sum(numpy.exp(v - x)))  
