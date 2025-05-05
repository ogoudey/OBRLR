import sys
import argparse
import yaml

import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--params', type=str, required=False, help="Parameters file")
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
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
    
    if args.test:
        composition = "reset_eef"
        policy = sac.load_saved_model(args.params, params["objective"]["move_eef"], "pi")
        sac.test(params["objective"]["move_eef"], composition, policy)
        
        composition = "midway_eef"        
        policy = sac.load_saved_model(args.params, params["objective"]["carry_cube"], "pi")
        sac.test(params["objective"]["carry_cube"], composition, policy)
        sys.exit()        
    
    
    objective_components = params["objective"]
    parameters_name = args.params.split('/')[1]
    policies = dict()
    for component in objective_components.keys():
        comp_params = objective_components[component]
        if "compositor" in comp_params.keys():
            composition = comp_params["compositor"]
        else:
            print("Learning", component, "with inputs", comp_params["pi"]["inputs"], "and outputs", comp_params["pi"]["outputs"])
            pi = sac.train(comp_params, composition, parameters_name)
            policies[component] = pi
    if not args.skip_test:
        if input("Test? (y/n): ") == "y":
            composition = "reset_eef"
            sac.test(objective_components["move_eef"], composition, policies["move_eef"])
            composition = "midway_eef"
            sac.test(objective_components["carry_cube"], composition, policies["carry_cube"])
    
    print("python done.")
