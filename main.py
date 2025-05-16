import sys
import argparse
import yaml
import os

import logging
logging.disable(logging.WARNING)


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--params', type=str, required=False, help="Parameters file")
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cyclic_test', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    args = parser.parse_args()
    
    
    # Load parameters file
    if not args.params:
        print("Please include --params <parameters>")
    try:
        with open(args.params + '.yaml', "r") as f:
            params = yaml.safe_load(f)
            parameters_name = args.params.split('/')[1]
        print("Loaded parameters")
    except Exception:
        try:
            with open("parameters/" + args.params + ".yaml", "r") as f:
                params = yaml.safe_load(f)
                parameters_name = args.params
        except Exception:
            try:
                with open(args.params, "r") as f:
                    params = yaml.safe_load(f)
                    parameters_name = args.params.remove_suffix(".yaml")
            except Exception:
                raise Exception("Rarameters file not found.")
    
    print(params["other_parameters"]["description"])
    
    import soft_actor_critic as sac # includes policy network

    objective = params["objective"]   
    
    inter_test_memory = None
    if args.test:
        print("Just testing...")
        for learned_component in objective["components"].keys():
            composition = objective["components"][learned_component]["composition"]
            component = objective["components"][learned_component]["component"]
            if "epilogue" in objective["components"][learned_component].keys():
                epilogue = objective["components"][learned_component]["epilogue"]
            else:
                epilogue = None
            policy = sac.load_pushed_policy(learned_component, component)
            inter_test_memory = sac.test(component, composition, policy, inter_test_memory, epilogue=epilogue)
        sys.exit()
    if args.cyclic_test:
        print("Cyclic test...")
        while True:
            for learned_component in objective["components"].keys():
                composition = objective["components"][learned_component]["composition"]
                component = objective["components"][learned_component]["component"]
                if "epilogue" in objective["components"][learned_component].keys():
                    epilogue = objective["components"][learned_component]["epilogue"]
                else:
                    epilogue = None
                policy = sac.load_pushed_policy(learned_component, component)
                inter_test_memory = sac.test(component, composition, policy, inter_test_memory, epilogue=epilogue)
        
    # Show committed policies


    
    policies = dict()
    for learning_component in objective["components"].keys():
        composition = objective["components"][learning_component]["composition"]
        component = objective["components"][learning_component]["component"]
        
        if sac.redundancy_check(learning_component):
            intfc = input("Still train? (y/n): ")
            if intfc == "n":
                continue
        print("Learning", learning_component, ":\nInputs:", component["pi"]["inputs"], "\nOutputs:", component["pi"]["outputs"], "\nComposition:", composition, "\nReward:", list(component["reward"].keys()))
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
                print(learned_component, "committed!")
                # Deal with merges.
            else:
                print(learned_component, "skipped.")

    if sac.push():
        print("Policies pushed.")
    else:
        print("Committed policies not pushed.")
    
                
                
    # Ask for push into sim_policies
    print("python done.")
