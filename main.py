
import argparse
import yaml


if __name__ == "__main__":
    
    with open('params.yaml', "r") as f: #file name is variable arg
        params = yaml.safe_load(f)
    print(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    
    if args.real:
        import real # irl robot stuff
        # unused in current algorithm
        
    # later: check if already trained
    
    #___Simulated stuff___:
    import interface
    sim = interface.Sim()
    
    import soft_actor_critic as sac # includes policy network


    # test SAR
    
    #sac.test_single_SAR(sim)
    #    
    
    sac.train(sim, params["training_parameters"])

    sac.test(sim) # really "watch"
    
    #


        
