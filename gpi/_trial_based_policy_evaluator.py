from _base import GeneralPolicyIterationComponent
from mdp import ClosedFormMDP
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None
        ):
        super().__init__()
        self.trial_interface = trial_interface
        self.gamma = gamma
        self.max_trial_length = max_trial_length
        self.exploring_starts = exploring_starts
        self.random_state = random_state

        print("Gamma ",gamma)
        print("Max Trial Length ", max_trial_length)
        print("Explorint Starts ", bool)


    
    def step(self):
        """
            creates and processes a trial to update state-values and q-values
        """
        q = {}
        v = {}
        states = []
        s = None
        N = pd.DataFrame(columns=['s','a','r'])
        #df.loc[len(df)] = ["s0", "a0", "s1", 1]
        # df = pd.concat([df, new_row], ignore_index=True)s
        

        if self.exploring_starts:
            s, r = self.trial_interface.get_random_state()
        else:
            s,r = self.trial_interface.draw_init_state()


        actions = self.trial_interface.get_actions_in_state(s)
        
        rn = self.random_state.random_integers(0,len(actions)-1)
        a = actions[rn]

        m0 = pd.DataFrame({"s":s,"a":a,"r":r})
        N = pd.concat([N,m0], ignore_index=True)

        

        for i in range(1,self.max_trial_length): 
            s, r = self.trial_interface.exec_action(s,a)
            moment = pd.DataFrame({"s":next_s,'a':a,'r':r})
            N = pd.concat([N,moment], ignore_index=True)


        self.trial_interface.__class__
        
    
    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError