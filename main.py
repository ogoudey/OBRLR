
import argparse
import yaml

import logging
logging.disable(logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--params', type=str, required=False, help="Parameters file")
    parser.add_argument('--pi', type=str, required=False, help="Pre-trained policy")
    parser.add_argument('--q1', type=str, required=False, help="Pre-trained critic1")
    parser.add_argument('--q2', type=str, required=False, help="Pre-trained critic2")
    parser.add_argument('--rb', type=str, required=False, help="Existing replay buffer")
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    
    
    # Load parameters file
    try:
        with open(args.params, "r") as f:
            params = yaml.safe_load(f)
        print("Loaded parameters")
    except Exception:
        raise Exception("Missing parameters file.")
        
    
    print(params["other_parameters"]["description"])
    
    import soft_actor_critic as sac # includes policy network
    
    if args.real:
        from real import real # irl robot stuff
        # unused in current algorithm
        if not args.pi:
            print("Please provide a policy.")
            return 0
        policy = sac.load_saved_policy(args.pi)
        real.test_policy(policy)
        
        return 1

    import interface
    sim = interface.Sim(params["training_parameters"]["reward_function"])
    
    
    

    sac.train(sim, params["training_parameters"], args)

    sac.test(sim)
    print("Done.")
