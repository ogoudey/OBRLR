
import argparse
import yaml

import logging
logging.disable(logging.INFO)
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
    
    if args.real:
        from real import real # irl robot stuff
        # unused in current algorithm

    import interface
    sim = interface.Sim(params["training_parameters"])
    
    import soft_actor_critic as sac # includes policy network
    
    if "train2" in params["training_parameters"].keys():
        sac.train2(sim, params["training_parameters"], args)
    else:
        sac.train(sim, params["training_parameters"], args)
    if "testing" in params["training_parameters"].keys():
        sac.test(sim)
    print("python done.")
