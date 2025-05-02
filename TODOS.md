# TODOs

Initiate sub-goal paradigm:

    build networks off parameters    
    hindsight only works in the general case
    
    1. Path planning
        - generalized reward function (taking parameters as input, or make reward relative always)
        - done=1 on goal1
        - lock grasp to wide
    2. Grasping
        - combine with 1 or 3?
    3. Lift
        - adding cube goal position
        

Example parameters file
    Change the network construction or just set initial weights to 0? (What happens if the latter?)
    parts:
        goal1
        learning_goal1:
            pi:
                inputs:
                    eef_pos
                    eef_delta
                    eef_goal_pos
                outputs:
                    eef_desired_move
        goal2
        learning_goal2:
            pi:
                inputs:
                    eef_pos #need?
                    eef_cube_pos
                    delta
                    goal_cube 
                    
                    
                    

Is there a way to logically say what's in these files? - Planning, FUTURE PROJECT**
** and there's an executive branch to the biological reward system...             
                    
Executor:
    generalized network or (simpler) loading all beforehand and then sequencing them based on done

Saved (task = the only task):
    tasks/
        task_name/
            goalI_{success/failure}
            goalN
            
make form_state take params
random note: the Q-network just extends the state by the action
can we really say what should be tracked in these sub-goal problems?
    

        
    
