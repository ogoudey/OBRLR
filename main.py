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
    
    
    """
    if args.test:
        if "lift" in params["objective"].keys():
            composition = "overall"
            policy = sac.load_saved_model(args.params, params["objective"]["lift"], "pi")
            sac.test(params["objective"]["lift"], composition, policy)
        if "reset_eef" in params["objective"].keys():
            composition = "reset_eef"
            policy = sac.load_saved_model(args.params, params["objective"]["move_eef"], "pi")
            sac.test(params["objective"]["move_eef"], composition, policy)
        
        if "carry_cube" in params["objective"].keys():
            composition = "midway_eef"        
            policy = sac.load_saved_model(args.params, params["objective"]["carry_cube"], "pi")
            sac.test(params["objective"]["carry_cube"], composition, policy)
        sys.exit()
    """
    # Show committed policies

    objective = params["objective"]
    parameters_name = args.params.split('/')[1]
    policies = dict()
    for learning_component in objective["components"].keys():
        composition = objective["components"][learning_component]["composition"]
        component = objective["components"][learning_component]["component"]
        
        if sac.redundancy_check(component):
            intfc = input("Policy for " + learning_component + " already pushed. Skip?(y/n): ")
            if intfc == "y":
                continue
        print("Learning", learning_component, "with inputs", component["pi"]["inputs"], "and outputs", component["pi"]["outputs"])
        pi = sac.train(component, composition, parameters_name)
        policies[learning_component] = pi
        

    if input("Test? (y/n): ") == "y":
        for learned_component in policies.keys():
            composition = objective["components"][learned_component]["composition"]
            component = objective["components"][learned_component]["component"]

            print("Testing", learned_component, "with inputs:\n\t", component["pi"]["inputs"], "and outputs:\n\t", component["pi"]["outputs"])
            sac.test(component, composition, policies[learned_component])
            success = sac.commit(learned_component, pi)
            if success:
                print(learned_component, "added!")
                # Deal with merges.
            else:
                print(learned_component, "skipped.")
                
                
    # Ask for push into sim_policies
    print("python done.")
