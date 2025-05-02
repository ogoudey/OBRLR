
import argparse
import yaml

import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--params', type=str, required=False, help="Parameters file")
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    
    
    # Load parameters file
    try:
        with open(args.params + '.yaml', "r") as f:
            params = yaml.safe_load(f)
        print("Loaded parameters")
    except Exception:
        raise Exception("Missing parameters file.")
        
    
    print(params["other_parameters"]["description"])
    
    import soft_actor_critic as sac # includes policy network
    
    logger = sac.setup_logger(params["training_parameters"], args.params)
    
    if args.real:
        from real import real # irl robot stuff
        # unused in current algorithm
        if not args.pi:
            print("Please provide a policy.")
            
        policy = sac.load_saved_policy(args.pi)
        real.test_policy(policy)
        
        

    import interface
    sim = interface.Sim(params["training_parameters"])
    
    
    

    pi = sac.train(sim, params["training_parameters"], args, logger)


    if "testing" in params["training_parameters"].keys():
        sac.test(sim, pi)
    print("python done.")
