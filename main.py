
import argparse
import yaml

import logging
logging.disable(logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--params', type=str, required=False, help="Parameters file")
    parser.add_argument('--model', type=str, required=False, help="Pre-trained model")
    parser.add_argument('--rb', type=str, required=False, help="Existing replay buffer")
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    
    
    # Load parameters file
    try:
        with open(args.params, "r") as f:
            params = yaml.safe_load(f)
        print("Loaded parameters")
    except Exception:
        print("Loading default parameters...")
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
    
    if args.real:
        from real import real # irl robot stuff
        # unused in current algorithm

    import interface
    sim = interface.Sim()
    
    import soft_actor_critic as sac # includes policy network
    
    if args.model:
        
        sac.load_saved_model(args.model)
        
        sac.test(sim)
        
    else:
        if args.rb:
            sac.train(sim, params["training_parameters"], args.rb)
        else:
            sac.train(sim, params["training_parameters"])

        sac.test(sim)       
