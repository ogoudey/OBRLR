
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
    parser.add_argument('--q', type=str, required=False, help="Pre-trained qnetwork")
    parser.add_argument('--rb', type=str, required=False, help="Existing replay buffer")
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    
    
    # Load parameters file
    try:
        with open(args.params, "r") as f:
            params = yaml.safe_load(f)
        print("Loaded parameters")
    except Exception:
        print(args.params, "no good. Loading default parameters...")
        with open("parameters/params.yaml", "r") as f:
            params = yaml.safe_load(f)
    
    if args.real:
        from real import real # irl robot stuff
        # unused in current algorithm

    import interface
    sim = interface.Sim()
    
    import soft_actor_critic as sac # includes policy network
    
    if args.pi:
        if args.q:
            if args.rb:
                sac.train(sim, params["training_parameters"], args.pi, args.q, args.rb)
            else:
                sac.train(sim, params["training_parameters"], args.pi, args.q)
        else:
            if args.rb:
                sac.train(sim, params["training_parameters"], args.pi, None, args.rb)
            else:
                sac.train(sim, params["training_parameters"], args.pi)

    else:
        if args.q:
            if args.rb:
                sac.train(sim, params["training_parameters"], None, args.q, args.rb)
            else:
                sac.train(sim, params["training_parameters"], None, args.q)
        else:
            if args.rb:
                sac.train(sim, params["training_parameters"], None, None, args.rb)
            else:
                sac.train(sim, params["training_parameters"])

    sac.test(sim)       
