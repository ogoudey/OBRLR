import argparse
import yaml

import logging
logging.disable(logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--params', type=str, required=False, help="Parameters file")
    parser.add_argument('--alg', type=str, default='ppo', help='Algorithm to use')
    parser.add_argument('--pi', type=str, required=False, help="Pre-trained policy")
    parser.add_argument('--q1', type=str, required=False, help="Pre-trained critic1")
    parser.add_argument('--q2', type=str, required=False, help="Pre-trained critic2")
    parser.add_argument('--rb', type=str, required=False, help="Existing replay buffer")
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--headless', action='store_true', help="Run without GUI (offscreen rendering only)")
    parser.add_argument('--mode', type=str, required=False, default="train", help="Existing replay buffer")

    args = parser.parse_args()
    
    
    # Load parameters file
    try:
        with open(args.params, "r") as f:
            params = yaml.safe_load(f)
        print("Loaded parameters")
    except Exception:
        print(args.params, "no good. Loading default parameters...")
        with open("parameters/ppo.yaml", "r") as f:
            params = yaml.safe_load(f)
    
    if args.real:
        from real import real # irl robot stuff
        # unused in current algorithm

    import sb_interface
    sim = sb_interface.Sim(params["training_parameters"])
    
    import soft_actor_critic as sac # includes policy network

    from ppo_training import train_ppo, test_ppo
    
    if args.mode == 'train':
        if args.alg == 'sac':
            sac.train(sim, params["training_parameters"], args)
        elif args.alg == 'ppo':
            train_ppo(sim, params["training_parameters"], args)
    else:
        test_ppo(sim)




    #sac.test(sim)
    print("Done.")
