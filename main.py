
import argparse


if __name__ == "__main__":

    
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
    
    sac.train(sim)

    sac.test(sim) # really "watch"
    
    #


        
