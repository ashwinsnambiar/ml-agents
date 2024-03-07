from typing import Any, Dict
import yaml

import optuna

#todo: multiple definition remove
N_EVALUATIONS = 4

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
        config_file.close()

    with open(path, 'w') as config_file:
        yaml.dump(data, config_file)
        config_file.close()


class TrialEvalCallback():
    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial
        self.eval_index = 1
        self._is_pruned = False
    
    def trial_eval_callback(self, curr_step_num: int, max_step_num: int, last_mean_reward: float):
        #called in rltrainer.py advance function.
        eval_freq = max_step_num // N_EVALUATIONS
        print("in trial eval")
        # checks if current step number is a multipple of the evaluation frequency
        # print("eval freg ", eval_freq)
        # print("curr step ", curr_step_num)
        # print("curr_step_num % eval_freq ", curr_step_num % eval_freq)
        if curr_step_num >= eval_freq * self.eval_index  and eval_freq > 0 and curr_step_num > 0:
            #sends intermediate result to optuna for pruning
            print("inside trial eval if")
            self.trial.report(last_mean_reward, self.eval_index)
            self.eval_index += 1
            if self.trial.should_prune():
                self._is_pruned = True
                print("in trial eval is pruned ", self._is_pruned)
        return self._is_pruned
    
    def is_pruned(self):
        print("is pruned in is pruned ", self._is_pruned)
        return self._is_pruned




# def objective(trial: optuna.Trial) -> float:
#     trainer_settings = unity_mlagents_param_sampler(trial)
    
#     #todo: assign the trainer settings to the config files
#     absPath = os.path.abspath(__file__)
#     fileDirectory = os.path.dirname(absPath)
#     configFileRelPath = os.path.join("config", "sia_20_2.yaml")
#     configFileAbsPath = os.path.join(fileDirectory, configFileRelPath)
#     update_config_file(configFileAbsPath)

#     #todo: call the learn.py or mlagents-learn with the cl arguments
#     path = "/home/ashwin/Coding/Project_Thesis/virero_unity22_linux/.venv/lib/python3.10/site-packages/mlagents/trainers/learn.py"
#     arg = "python " + path + " " + configFileAbsPath
#     os.system(arg)



#     return 0.0

# if __name__ == "__main__":
#     pass

# #todo: to add to main funtion
# def toadd():
#     # Select the sampler, can be random, TPESampler, CMAES, ...
#     sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
#     # Do not prune before 1/3 of the max budget is used
#     pruner = MedianPruner(
#         n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
#     )
#     # Create the study and start the hyperparameter optimization
#     study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

#     try:
#         study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
#     except KeyboardInterrupt:
#         pass

#     print("Number of finished trials: ", len(study.trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print(f"  Value: {trial.value}")

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")

#     print("  User attrs:")
#     for key, value in trial.user_attrs.items():
#         print(f"    {key}: {value}")

#     # Write report
#     study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

#     fig1 = plot_optimization_history(study)
#     fig2 = plot_param_importances(study)

#     fig1.show()
#     fig2.show()
