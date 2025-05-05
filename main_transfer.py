import soft_actor_critic as sac # includes policy network
import yaml
import utilities
import sys

def main():
    ## Load by parameters (they havetendency to get overriden)
    working_parameters = "parameters/standard_dense"
    with open(working_parameters + ".yaml", "r") as f:
        params = yaml.safe_load(f) 


    ## Safety Sim ###
    
    composition = "reset_eef"
    policy = sac.load_saved_model(working_parameters, params["objective"]["move_eef"], "pi")    
    sac.test(params["objective"]["carry_cube"], composition, policy)
    
    
    
    #############
       

    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        composition = "reset_eef"
        policy = sac.load_saved_model(working_parameters, params["objective"]["move_eef"], "pi")    
        sac.test(params["objective"]["carry_cube"], composition, policy, router)
        sys.exit()
        
if __name__ == "__main__":
    main()
