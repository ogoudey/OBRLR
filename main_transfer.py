import soft_actor_critic as sac # includes policy network
import yaml
import utilities
import sys

def main():
    ## Load by parameters (they havetendency to get overriden)
    working_parameters = "parameters/transferable"
    with open(working_parameters + ".yaml", "r") as f:
        params = yaml.safe_load(f) 

    policy_dir = "convergent_networks"
    ## Safety Sim ###
    print("Safety Sim")
    composition = "reset_eef"
    
    policy = sac.load_saved_model(working_parameters, params["objective"]["move_eef"], "pi")    
    sac.test(params["objective"]["move_eef"], composition, policy, router=None, cut_component=True)
    
    composition = "midway_eef"
    policy = sac.load_saved_model(working_parameters, params["objective"]["carry_cube"], "pi")    
    sac.test(params["objective"]["carry_cube"], composition, policy, router=None, cut_component=False)
    
    
    #############
       

    args = utilities.parseConnectionArguments()
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        composition = "reset_eef"
    
        policy = sac.load_saved_model(working_parameters, params["objective"]["move_eef"], "pi")    
        sac.test(params["objective"]["move_eef"], composition, policy, router)
        
        composition = "midway_eef"
        policy = sac.load_saved_model(working_parameters, params["objective"]["carry_cube"], "pi")    
        sac.test(params["objective"]["carry_cube"], composition, policy, router)
        
if __name__ == "__main__":
    main()
