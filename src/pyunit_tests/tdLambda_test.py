import unittest
from learning_toolkit import *
from verifier import *
class tdLambda_test(unittest.TestCase):
    """
    def test_OldTD_variablesUpdate(self):
        print "OLD TD theta"
        td = tdLambda()
        td.tdAgent([1,1], 1)
        print td.getTheta()
        td.tdAgent([1,1], 1)
        print td.getTheta()
        td.tdAgent([1,1], 1)
        print td.getTheta()
        td.tdAgent([1,1], 1)
        print td.getTheta()
        print "FINISHED \n"
    """
    def test_TD_Predictions(self):
        print "TD FOR PTL Predictions"
        td = TDLambdaLearner()
        td.update([1,1], 1)
        print td.predict([1,1])
        td.update([1,1], 1)
        print td.predict([1,1])
        td.update([1,1], 1)
        print td.predict([1,1])
        td.update([1,1], 1)
        print td.predict([1,1])
        print "FINISHED \n"
        
    def test_PTLTD(self):
        print "TD FOR PTL"
        td = TDLambdaLearner()
        td.update([1,1], 1)
        print td.theta
        td.update([1,1], 1)
        print td.theta
        td.update([1,1], 1)
        print td.theta
        td.update([1,1], 1)
        print td.theta
        print "FINISHED \n"
    
    def test_ptlNode(self):
        print "PTL NODE theta"
        ptn = Partition_Tree_Learner_Node(TDLambdaLearner, 1, None, False)
        ptn.update([1,1], 1)
        print ptn.learner.theta
        ptn.update([1,1], 1)
        print ptn.learner.theta
        ptn.update([1,1], 1)
        print ptn.learner.theta
        ptn.update([1,1], 1)
        print ptn.learner.theta
        print "FINISHED \n \n \n \n"
        
    def test_ptlNode_prediction(self):
        print "PTL NODE Prediction"
        ptn = Partition_Tree_Learner_Node(TDLambdaLearner, 1, None, False)
        ptn.update([1,1], 1)
        print ptn.predict([1,1])
        ptn.update([1,1], 1)
        print ptn.predict([1,1])
        ptn.update([1,1], 1)
        print ptn.predict([1,1])
        ptn.update([1,1], 1)
        print ptn.predict([1,1])
        print "FINISHED \n \n \n \n"
    
    def test_ptlNode_stepping(self):
        print "PTL NODE theta"
        ptn = Partition_Tree_Learner_Node(TDLambdaLearner, 1, None, False)
        ptn.update([1,1], 1)
        print ptn.learner.theta
        ptn.update([1,1], 1)
        print ptn.learner.theta
        ptn.update([1,1], 1)
        print ptn.learner.theta
        ptn.update([1,1], 1)
        print ptn.learner.theta
        ptn.update([1,1], 1)
        print ptn.learner.theta
        self.assertTrue(ptn.num_steps == 1, "Steps are not being counted properly")
        print "FINISHED \n \n \n \n"
    
    def test_PTL_Singleton(self):
        print "PTL SINGLETON learner predictions"
        ptl = Partition_Tree_Learner(1, TDLambdaLearner)
        ptl.update([1,1], 1)
        print ptl.get_learner_predictions([1,1])
        ptl.update([1,1], 1)
        print ptl.get_learner_predictions([1,1])
        ptl.update([1,1], 1)
        print ptl.get_learner_predictions([1,1])
        print "FINISHED \n \n \n \n"
    
    def test_PTL(self):
        print "PTL weights"
        ptl = Partition_Tree_Learner(7, TDLambdaLearner)
        ptl.update([1,1], 1)
        print ptl.get_learner_weighting()
        ptl.update([1,1], 1)
        print ptl.get_learner_weighting()
        ptl.update([1,1], 1)
        print ptl.get_learner_weighting()
        print "FINISHED \n \n \n \n"
     
    def test_PTL_withVerifier(self):
        print "PTL with verifier"
        ptl = Partition_Tree_Learner(8, TDLambdaLearner)
        node = ptl.nodes[0]
        print "Gamma " + str(node.learner.rlGamma) 
        ver = Verifier(node.learner.rlGamma)
        for item in range(500):
            ptl.update([1,1], 1)
            ver.updateReward(1)
        
        print str(ver.calculateReturn()) + " Actual Return"
        print str(ptl.predict([1,1])) + " Predicted Return"
        print "FINISHED \n \n \n \n"
     
    def test_true_online_td(self):
        print "TRUE ONLINE TD PREDICTIONS"
        td = True_Online_TD()
        td.update([1,1], 1)
        print td.predict([1,1])
        td.update([1,1], 1)
        print td.predict([1,1])
        td.update([1,1], 1)
        print td.predict([1,1])
        td.update([1,1], 1)
        print td.predict([1,1])
        print "FINISHED \n \n \n \n"
        
    def test_true_online_td_weights(self):
        print "TRUE ONLINE TD WEIGHTS"
        td = True_Online_TD()
        td.update([1,1], 1)
        print td.theta
        td.update([1,1], 1)
        print td.theta
        td.update([1,1], 1)
        print td.theta
        td.update([1,1], 1)
        print td.theta
        print "FINISHED \n \n \n \n"
        
    def test_true_online_td_weights_longtime(self):
        print "TRUE ONLINE TD WEIGHTS"
    
        td = True_Online_TD()
        for i in range(100):
            td.update([1,1], 1)
            print td.predict([1,1])
      
        print "\n \n \n \n"
      
        td = TDLambdaLearner()
        for i in range(60):
            td.update([1,1], 1)
            print td.predict([1,1])
            
        print "FINISHED \n \n \n \n"
    
if __name__ == '__main__':
    unittest.main()
#