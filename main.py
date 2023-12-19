if __name__ == "__main__":
    import random
    import numpy as np
    import torch as T
    import warnings
    from muavenv import Environment
    from muavenv.global_vars import *
    from muavenv.utils import *
    from parsers.scenario import *
    from parsers.training import *
    from muavenv.global_vars import *
    from train_and_sweep import *
    import gym

    # Use the seeds set in 'global_vars.py':
    random.seed(SEED)
    np.random.seed(SEED)
    # If you want to use a random seed for the weights initialization, then comment the following two lines:
    if SEED!=None:
        T.manual_seed(SEED)
    
    # Uncomment the following line when a CUDA error is thrown, and you want to be sure about the line that is throwing the error:
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    # Terminal arguments parsing:
    main_aux = MainAux()
    args = main_aux.terminal_parser()
    logs_dir, model_save_dir, model_load_dir, plot_dir, settings_src_dir, settings_dst_dir, external_obs_dir = main_aux.folders_generation(args=args)

    # Import scenario and training parameters:
    scnr_cfg = ScenarioConfig(settings_src=settings_src_dir, settings_dst=settings_dst_dir)
    train_cfg = TrainingConfig(settings_src=settings_src_dir, settings_dst=settings_dst_dir, algorithm=main_aux.algorithm_folder)

    # Run a server side on a different terminal only if --external_communication is enabled:
    if args.external_communication:
        from subprocess import call
        server_running_cmd = 'python3 ' + PATH_FOR_EXT_COMMUNICATION + 'server.py'
        call(['gnome-terminal', '--', 'sh', '-c', server_running_cmd + '; bash'])

    # Init environment
    env = Environment(args=args, scnr_cfg=scnr_cfg, train_cfg=train_cfg, train_phase=args.train, FIR_file=args.FIR_file, FIR_ID=args.FIR_ID, external_communication=args.external_communication, algorithm=main_aux.algorithm_folder)
    
    # Enabling/disabling rendering, debugging features and number of rendered steps to save for each animation:
    visible = False # True
    debug = False # True
    render_steps_saving = 30 #20

    # Writer for Tensorboard:
    writer = SummaryWriter(log_dir=logs_dir)

    # Prepare the env, args and configuration variables before running:
    info_ready = TrainANDSweep(env=env,
                               args=args,
                               scnr_cfg=scnr_cfg,
                               train_cfg=train_cfg,
                               writer=writer,
                               model_save_dir=model_save_dir,
                               model_load_dir=model_load_dir,
                               plot_dir=plot_dir,
                               settings_src_dir=settings_src_dir,
                               external_obs_dir=external_obs_dir,
                               debug=debug,
                               visible=visible,
                               render_steps_saving=render_steps_saving)

    # Run:
    info_ready.run()