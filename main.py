
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
    
    objective_components = params["objective"]
    policies = dict()
    for component in objective_components.keys():
        
        if "compositor" in objective_components[component].keys():
            composition = objective_components[component]["compositor"]
        else:
            print("Learning", component, "...")
            pi = sac.train(objective_components[component], composition)
            policies[component] = pi
    
    if input("Test? (y/n): ") == "y":
        for component in policies.keys():
            sac.test([component], composition, policies[component])

    
    print("python done.")
