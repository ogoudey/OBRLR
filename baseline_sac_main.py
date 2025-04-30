import argparse
import yaml
import logging
import warnings

logging.disable(logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC with HER on Kinova3")
    parser.add_argument(
        '--params', type=str, required=False, help="path to yaml file with params")
    parser.add_argument(
        '--model', type=str, required=False, help="path to pre-trained model")
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
    
    # Initialize the simulation
    import interface
    sim = interface.Sim(params)
    
    # Train sac with her using the loaded parms
    import baseline3_sac
    training_params = params.get("training_parameters", params)
    model = baseline3_sac.train(
        sim, training_params, args)