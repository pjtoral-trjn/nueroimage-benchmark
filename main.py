import sys
import argparse
# from config.config import experiment_config
# from model.Model import Model

def config_parser():
    configuration_parser = argparse.ArgumentParser()
    # defaults
    configuration_parser.add_argument("-tls", "--trainable_layers", type=int, default=1, help="Trainable Layers")
    configuration_parser.add_argument("-e", "--epochs", type=int, default=50, help="Training Epochs")
    configuration_parser.add_argument("-s", "--seed", type=int, default=5, help="Randomizer Seed")
    configuration_parser.add_argument("-tc", "--target_column", type=str, default="label", help="Target Column")
    configuration_parser.add_argument("-vc", "--volume_column", type=str, default="volume", help="Volume Column")
    configuration_parser.add_argument("-o", "--optimizer", type=str, default="adamw", help="Model Optimizer")
    configuration_parser.add_argument("-pt", "--pre_train", type=bool, default=False, help="Is this experiment pretraining?")

    # required configuration
    configuration_parser.add_argument("-g","--gpu", type=int, help="GPU ID Selection")
    configuration_parser.add_argument("-bs", "--batch_size", type=int, help="Batch Size")
    configuration_parser.add_argument("-ma", "--model_architecture", type=str, help="Model Architecture Selection")
    configuration_parser.add_argument("-l", "--loss", type=str, help="Model Loss")
    configuration_parser.add_argument("-en", "--experiment_name", type=str, help="Experiment Name, for storage")
    configuration_parser.add_argument("-trp", "--train_pathway", type=str, help="Train Pathway CSV")
    configuration_parser.add_argument("-tep", "--test_pathway", type=str, help="Test Pathway CSV")
    configuration_parser.add_argument("-ilr", "--init_learning_rate", type=float, help="Initial Learning Rate for the optimizer")

    return configuration_parser

if __name__ == '__main__':
    parser = config_parser()
    # configuration = sys.argv[1]
    print("----- Experiment Begin -----")
    print("Running the following configuration:")
    print(parser)
    # experiment_model = Model(experiment_config[configuration])
    # experiment_model.run_pipeline()
    print("----- Experiment Complete -----")



