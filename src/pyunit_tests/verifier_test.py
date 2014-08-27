import unittest
from learning_toolkit import *
from verifier import *

class tdLambda_test(unittest.TestCase):
        """  
        def test_verifier_history_basic(self):
            print "VERIFIER TEST"
            v = Verifier(0.01)
            v.updateReward(1)
            print v.rewardHistory
            v.updateReward(2)
            print v.rewardHistory
            
        def test_verifier_history_complex(self):
            print "VERIFIER TEST"
            v = Verifier(0.5)
            print v.horizon
            v.updateReward(1)
            print v.rewardHistory
            v.updateReward(2)
            print v.rewardHistory
            v.updateReward(3)
            print v.rewardHistory
            v.updateReward(4)
            print v.rewardHistory
            v.updateReward(5)
            print v.rewardHistory
            v.updateReward(6)
            print v.rewardHistory
            v.updateReward(7)
            print v.rewardHistory
            v.updateReward(8)
            print v.rewardHistory
            
        def test_verifier_history_return_complex(self):
            print "VERIFIER TEST"
            v = Verifier(0.5)
            v.updateReward(1)
            print v.calculateReturn()
            v.updateReward(1)
            print v.calculateReturn()
            v.updateReward(1)
            print v.calculateReturn()
            v.updateReward(1)
            print v.calculateReturn()
            v.updateReward(1)
            print v.calculateReturn()
            v.updateReward(1)
            print v.calculateReturn()
            v.updateReward(1)
            print v.calculateReturn()
            v.updateReward(1)
            print v.calculateReturn()
        """    
        
        def test_verifier_history_return_vs_TrueOnlineTD_complex(self):
            """
            Q blew up, check what's up with that.
            """
            # TRUE ONLINE PRINTS 0.0 and TD PRINTS 10
            print "VERIFIER TEST VS TD TRUE"
            #td = True_Online_TD(1, 3, 0.5, 0.9, 0.9, 2**2)
            td = True_Online_TD2(1, 3, 0.5, 0.9, 0.9, 2**(2*1))
            verifier = Verifier(td.rlGamma)
            #td = TDLambdaLearner(1, 3, 0.01, 0.5, 0.9, 0.9, 2**(2*1))
            print "Tiles: " + str(td.numTilings)
            print "parameters: " + str(td.parameters)
            
            for i in range(60):
                td.update([0,0,0], 1)
                verifier.updateReward(1)
                print "Verifier: " + str(verifier.calculateReturn())
                print "Prediction TD:  " + str(td.prediction)
                
        