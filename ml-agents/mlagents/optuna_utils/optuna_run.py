from typing import Any, Dict
import yaml

import optuna

#todo: multiple definition remove
N_EVALUATIONS = 5

def unity_mlagents_param_sampler(trial: optuna.Trial) -> Dict[str, Any]:
    batch_size = 2 ** trial.suggest_int("batch_size", 5, 10)
    buffer_size = batch_size * trial.suggest_int("buffer_size", 5, 20, step=3)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1, log=True)
    num_epoch = trial.suggest_int("num_epoch", 3, 8)
    return {
        'batch_size' : batch_size,
        'buffer_size' : buffer_size,
        'learning_rate' : learning_rate,
        'num_epoch' : num_epoch
    }


def update_config_file(trial: optuna.Trial, path):
    hyperparameters_dict = unity_mlagents_param_sampler(trial)
    # hyperparameters_dict = {
    #     'batch_size' : 128,
    #     'buffer_size' : 2048,
    #     'learning_rate' : 0.0003
    # }
    with open(path, "r") as config_file:
        data = yaml.load(config_file, Loader=yaml.FullLoader)
        hyperparameters = data['behaviors']['TouchCube']['hyperparameters']
        hyperparameters['batch_size'] = hyperparameters_dict["batch_size"]
        hyperparameters['buffer_size'] = hyperparameters_dict['buffer_size']
        hyperparameters['learning_rate'] = hyperparameters_dict['learning_rate']
        config_file.close()

    #todo: stop writing it alphabetically

    with open(path, 'w') as config_file:
        yaml.dump(data, config_file, sort_keys=False)
        config_file.close()


class TrialEvalCallback():
    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial
        self.eval_index = 1
        self._is_pruned = False
    
    def trial_eval_callback(self, curr_step_num: int, max_step_num: int, last_mean_reward: float):
        #called in rltrainer.py advance function.
        eval_freq = max_step_num // N_EVALUATIONS
        # print("in trial eval")
        # checks if current step number is a multipple of the evaluation frequency
        # print("eval freg ", eval_freq)
        # print("curr step ", curr_step_num)
        # print("curr_step_num % eval_freq ", curr_step_num % eval_freq)
        if curr_step_num >= eval_freq * self.eval_index  and eval_freq > 0 and curr_step_num > 0:
            #sends intermediate result to optuna for pruning
            # print("inside trial eval if")
            self.trial.report(last_mean_reward, self.eval_index)
            self.eval_index += 1
            if self.trial.should_prune():
                self._is_pruned = True
                # print("in trial eval is pruned ", self._is_pruned)
        return self._is_pruned
    
    def is_pruned(self):
        # print("is pruned in is pruned ", self._is_pruned)
        return self._is_pruned
