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
        self.q = {}
        self.v = {}
    
    def step(self):
        """
            creates and processes a trial to update state-values and q-values
        """

        rewards = {}
        policy = {}
        trial = pd.DataFrame(columns=['actual_state','action','reward','next_state'])
        
        if self.exploring_starts:
            s =self.trial_interface.get_random_state()
        else:
            s = self.trial_interface.draw_init_state()

        actions = self.trial_interface.get_actions_in_state(s)
        rn = self.random_state.random_integers(0,len(actions)-1)
        a = actions[rn]

        policy[s] = a

        s_next,r = self.trial_interface.exec_action(s,a)

        rewards[s] = r
        new_row = pd.DataFrame(data={"actual_state":s,"action": a,'reward':r,'next_state':s_next})
        trial = pd.concat([trial,new_row], ignore_index=True)

        # Se comienza a hacer el trial desde 1 hasta T
        for _ in range(1,self.max_trial_length+1):
            s = s_next

            if s not in policy.keys():
                actions = self.trial_interface.get_actions_in_state(s)
                rn = self.random_state.random_integers(0,len(actions)-1)
                a = actions[rn]

                policy[s] = a
            else:
                a = policy[s]
            s_next,r = self.trial_interface.exec_action(s,a)

            rewards[s] = r
            new_row = pd.DataFrame(data={"actual_state":s,"action": a,'reward':r,'next_state':s_next})
            trial = pd.concat([trial,new_row], ignore_index=True)

            
            

    
    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError