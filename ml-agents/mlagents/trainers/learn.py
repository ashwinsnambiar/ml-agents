# # Unity ML-Agents Toolkit
import pickle
from mlagents import torch_utils
import yaml
import time

import os
import numpy as np
import json

from typing import Callable, Optional, List

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from mlagents.optuna_utils.optuna_run import update_config_file, TrialEvalCallback
from mlagents.trainers.cli_utils import StoreConfigFile

import mlagents.trainers
import mlagents_envs
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.directory_utils import (
    validate_existing_directories,
    setup_init_path,
)
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.cli_utils import parser
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.settings import RunOptions

from mlagents.trainers.training_status import GlobalTrainingStatus
from mlagents_envs.base_env import BaseEnv
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.timers import (
    hierarchical_timer,
    get_timer_tree,
    add_metadata as add_timer_metadata,
    reset_timers
)
from mlagents_envs import logging_util
from mlagents.plugins.stats_writer import register_stats_writer_plugins
from mlagents.plugins.trainer_type import register_trainer_plugins

logger = logging_util.get_logger(__name__)

#constants for the optimisation run
#todo: edit the constants. also N_EVALUATIONS = 2 define in optuna.py too
N_TRIALS = 1000  # Maximum number of trials
#TIMEOUT = int(60 * 15)  # 15 minutes
N_JOBS = 1 # Number of jobs to run in parallel
SHOW_PROGRESS_BAR = False
N_STARTUP_TRIALS = 5 # Number of trials where pruning is skipped initially
N_EVALUATIONS = 10  # Number of evaluations during the training
N_WARMUP_STEPS = 4 # Here, steps indicate the number of pruning evaluation steps in my implementation as I am reporting the eval_index instead of step in RL_Trainer.py
# N_EVAL_ENVS = 5
N_MIN_TRIALS = 5 # Min number of trials required at a step to prune
TRAINING_STATUS_FILE_NAME = "training_status.json"

start_time = time.perf_counter()

def get_version_string() -> str:
    return f""" Version information:
  ml-agents: {mlagents.trainers.__version__},
  ml-agents-envs: {mlagents_envs.__version__},
  Communicator API: {UnityEnvironment.API_VERSION},
  PyTorch: {torch_utils.torch.__version__}"""


def parse_command_line(
    argv: Optional[List[str]] = None,
):
    _, _ = register_trainer_plugins()
    args = parser.parse_args(argv)
    return args


def run_training(
        run_seed: int, 
        options: RunOptions, 
        num_areas: int, 
        trial_eval: TrialEvalCallback = None
        ) -> None:
    """
    Launches training session.
    :param run_seed: Random seed used for training.
    :param num_areas: Number of training areas to instantiate
    :param options: parsed command line arguments
    """
    with hierarchical_timer("run_training.setup"):
        torch_utils.set_torch_config(options.torch_settings)
        checkpoint_settings = options.checkpoint_settings
        env_settings = options.env_settings
        engine_settings = options.engine_settings

        run_logs_dir = checkpoint_settings.run_logs_dir
        port: Optional[int] = env_settings.base_port
        # Check if directory exists
        validate_existing_directories(
            checkpoint_settings.write_path,
            checkpoint_settings.resume,
            checkpoint_settings.force,
            checkpoint_settings.maybe_init_path,
        )
        # Make run logs directory
        os.makedirs(run_logs_dir, exist_ok=True)
        # Load any needed states in case of resume
        if checkpoint_settings.resume:
            GlobalTrainingStatus.load_state(
                os.path.join(run_logs_dir, "training_status.json")
            )
        # In case of initialization, set full init_path for all behaviors
        elif checkpoint_settings.maybe_init_path is not None:
            setup_init_path(options.behaviors, checkpoint_settings.maybe_init_path)

        # Configure Tensorboard Writers and StatsReporter
        stats_writers = register_stats_writer_plugins(options)
        for sw in stats_writers:
            StatsReporter.add_writer(sw)

        if env_settings.env_path is None:
            port = None
        env_factory = create_environment_factory(
            env_settings.env_path,
            engine_settings.no_graphics,
            engine_settings.no_graphics_monitor,
            run_seed,
            num_areas,
            env_settings.timeout_wait,
            port,
            env_settings.env_args,
            os.path.abspath(run_logs_dir),  # Unity environment requires absolute path
        )

        env_manager = SubprocessEnvManager(env_factory, options, env_settings.num_envs)
        env_parameter_manager = EnvironmentParameterManager(
            options.environment_parameters, run_seed, restore=checkpoint_settings.resume
        )

        trainer_factory = TrainerFactory(
            trainer_config=options.behaviors,
            output_path=checkpoint_settings.write_path,
            train_model=not checkpoint_settings.inference,
            load_model=checkpoint_settings.resume,
            seed=run_seed,
            param_manager=env_parameter_manager,
            init_path=checkpoint_settings.maybe_init_path,
            multi_gpu=False,
        )
        # Create controller and begin training.
        tc = TrainerController(
            trainer_factory,
            checkpoint_settings.write_path,
            checkpoint_settings.run_id,
            env_parameter_manager,
            not checkpoint_settings.inference,
            run_seed,
        )

    # Begin training
    try:
        tc.start_learning(env_manager, trial_eval)
    finally:
        env_manager.close()
        write_run_options(checkpoint_settings.write_path, options)
        write_timing_tree(run_logs_dir)
        write_training_status(run_logs_dir)


def write_run_options(output_dir: str, run_options: RunOptions) -> None:
    run_options_path = os.path.join(output_dir, "configuration.yaml")
    try:
        with open(run_options_path, "w") as f:
            try:
                yaml.dump(run_options.as_dict(), f, sort_keys=False)
            except TypeError:  # Older versions of pyyaml don't support sort_keys
                yaml.dump(run_options.as_dict(), f)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save configuration to {run_options_path}. Make sure the directory exists"
        )


def write_training_status(output_dir: str) -> None:
    GlobalTrainingStatus.save_state(os.path.join(output_dir, TRAINING_STATUS_FILE_NAME))


def write_timing_tree(output_dir: str) -> None:
    timing_path = os.path.join(output_dir, "timers.json")
    try:
        with open(timing_path, "w") as f:
            json.dump(get_timer_tree(), f, indent=4)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save to {timing_path}. Make sure the directory exists"
        )


def create_environment_factory(
    env_path: Optional[str],
    no_graphics: bool,
    no_graphics_monitor: bool,
    seed: int,
    num_areas: int,
    timeout_wait: int,
    start_port: Optional[int],
    env_args: Optional[List[str]],
    log_folder: str,
) -> Callable[[int, List[SideChannel]], BaseEnv]:
    def create_unity_environment(
        worker_id: int, side_channels: List[SideChannel]
    ) -> UnityEnvironment:
        # Make sure that each environment gets a different seed
        env_seed = seed + worker_id
        return UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            seed=env_seed,
            num_areas=num_areas,
            no_graphics=no_graphics,
            no_graphics_monitor=no_graphics_monitor,
            base_port=start_port,
            additional_args=env_args,
            side_channels=side_channels,
            log_folder=log_folder,
            timeout_wait=timeout_wait,
        )

    return create_unity_environment


def run_cli(options: RunOptions, trial_eval: TrialEvalCallback = None) -> None:
    try:
        # print("inside run_cli ", trial_eval.trial.number)
        print(
            """
            ┐  ╖
        ╓╖╬│╡  ││╬╖╖
    ╓╖╬│││││┘  ╬│││││╬╖
 ╖╬│││││╬╜        ╙╬│││││╖╖                               ╗╗╗
 ╬╬╬╬╖││╦╖        ╖╬││╗╣╣╣╬      ╟╣╣╬    ╟╣╣╣             ╜╜╜  ╟╣╣
 ╬╬╬╬╬╬╬╬╖│╬╖╖╓╬╪│╓╣╣╣╣╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╒╣╣╖╗╣╣╣╗   ╣╣╣ ╣╣╣╣╣╣ ╟╣╣╖   ╣╣╣
 ╬╬╬╬┐  ╙╬╬╬╬│╓╣╣╣╝╜  ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╣╙ ╙╣╣╣  ╣╣╣ ╙╟╣╣╜╙  ╫╣╣  ╟╣╣
 ╬╬╬╬┐     ╙╬╬╣╣      ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣     ╣╣╣┌╣╣╜
 ╬╬╬╜       ╬╬╣╣      ╙╝╣╣╬      ╙╣╣╣╗╖╓╗╣╣╣╜ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣╦╓    ╣╣╣╣╣
 ╙   ╓╦╖    ╬╬╣╣   ╓╗╗╖            ╙╝╣╣╣╣╝╜   ╘╝╝╜   ╝╝╝  ╝╝╝   ╙╣╣╣    ╟╣╣╣
   ╩╬╬╬╬╬╬╦╦╬╬╣╣╗╣╣╣╣╣╣╣╝                                             ╫╣╣╣╣
      ╙╬╬╬╬╬╬╬╣╣╣╣╣╣╝╜
          ╙╬╬╬╣╣╣╜
             ╙
        """
        )
    except Exception:
        print("\n\n\tUnity Technologies\n")
    print(get_version_string())

    if options.debug:
        log_level = logging_util.DEBUG
    else:
        log_level = logging_util.INFO

    logging_util.set_log_level(log_level)

    logger.debug("Configuration for this run:")
    logger.debug(json.dumps(options.as_dict(), indent=4))

    # Options deprecation warnings
    if options.checkpoint_settings.load_model:
        logger.warning(
            "The --load option has been deprecated. Please use the --resume option instead."
        )
    if options.checkpoint_settings.train_model:
        logger.warning(
            "The --train option has been deprecated. Train mode is now the default. Use "
            "--inference to run in inference mode."
        )

    run_seed = options.env_settings.seed
    num_areas = options.env_settings.num_areas

    # Add some timer metadata
    add_timer_metadata("mlagents_version", mlagents.trainers.__version__)
    add_timer_metadata("mlagents_envs_version", mlagents_envs.__version__)
    add_timer_metadata("communication_protocol_version", UnityEnvironment.API_VERSION)
    add_timer_metadata("pytorch_version", torch_utils.torch.__version__)
    add_timer_metadata("numpy_version", np.__version__)

    if options.env_settings.seed == -1:
        run_seed = np.random.randint(0, 10000)
        logger.debug(f"run_seed set to {run_seed}")
    run_training(run_seed, options, num_areas, trial_eval)

def objective(trial: optuna.Trial, args) -> float:
    # Updating the config file with the current trial hyperparamters
    update_config_file(trial, StoreConfigFile.trainer_config_path)
    trial_eval = TrialEvalCallback(trial)

    #define start time
    start_time = time.perf_counter()


    #todo: how to initialize the first run with given config file parameters?? 
    
    nan_encountered = False
    try:
        options = RunOptions.from_argparse(args)
        # print(trial._trial_id)
        # print(trial.number)
        run_id = [options.checkpoint_settings.run_id, str(trial.number)]
        options.checkpoint_settings.run_id = '_'.join(run_id)
        
        # print(options.checkpoint_settings.run_id) 
        # Return execution to learn.py and continue the training with trial hyperparameters
        run_cli(options, trial_eval)
        reset_timers()
    except AssertionError as error:
        # Sometimes, random hyperparams can generate NaN
        print(error)
        nan_encountered = True

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")
    # print("before should prune in objective")
    # if trial_eval.is_pruned():
    #     # print(trial_eval.is_pruned())
    #     # print("in should prune in objective")
    #     raise optuna.exceptions.TrialPruned()
    # print("after should prune in objective")
    print(f"{options.checkpoint_settings.run_id} is {trial.user_attrs['last_mean_reward']}")
    elapsed_time = time.perf_counter() - start_time
    return trial.user_attrs['last_mean_reward'], elapsed_time


def start_optuna_tuning(args):
    """
    Takes the parsed command line arguments as input and starts the optuna study.
    It tries to maximise the mean reward of the trial runs by optimising the pre-defined 
    hyperparameters
    """
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_WARMUP_STEPS, 
        n_min_trials=N_MIN_TRIALS
    )
    storage_url = "sqlite:///results/opt1/trial1.db"
    
    # For restoring the sampler if initialised with a seed
    if os.path.isfile("results/opt1/sampler.pkl"):
        sampler = pickle.load(open("results/opt1/sampler.pkl", "rb"))

    study_name = "PPO_Hyperparameters"
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(storage=storage_url, sampler=sampler, pruner=pruner, 
                                study_name=study_name, direction="maximize", 
                                load_if_exists=True)
    
    # #Pruning not supported for multi objective funtions
    # study = optuna.create_study(storage=storage_url, sampler=sampler,
    #                         study_name=study_name, directions=['maximize', 'minimize'], 
    #                         load_if_exists=True)
    # to retry the failed trials
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.FAIL: 
            study.enqueue_trial(trial.params)

    try:
        study.optimize(
            lambda trial: objective(trial, args), 
            n_trials=N_TRIALS, 
            n_jobs=N_JOBS,
            show_progress_bar=SHOW_PROGRESS_BAR
            )
    except KeyboardInterrupt:
        # saving sampler, to restore later if needed. Code to be added later.
        with open("results/opt1/sampler.pkl", 'wb') as fout:
            pickle.dump(study.sampler, fout)
        pass


    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    #todo: Write report 
    #study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()


def main():
    # print("In main()")
    args = parse_command_line()
    # print("returned to main()")
    if args.optuna_tuning:
        start_optuna_tuning(args)
    else:
        run_cli(RunOptions.from_argparse(args))


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
