# Argument handling
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-r", "--run", action="store_true",
                    help="Start the learning process")

parser.add_argument("-tr", "--train_memories", type=int, default=50,
                    help="Number of memories you want to create")

parser.add_argument("-v", "--val_memories", type=int, default=20,
                    help="Number of validation memories you want to create")

parser.add_argument("-te", "--test_episodes", type=int, default=100,
                    help="Number of episodes to test network")

parser.add_argument("-e", "--environment", type=str, default='forest',
                    choices=["forest", "forest_houses", "forest_river", "forest_houses_river"],
                    help="Environment in which you train your agent (containing only forest or also houses and a river)")

parser.add_argument("-s", "--size", type=int, default='10',
                    help="Use the size of the map that you want to use")

parser.add_argument("-t", "--type", type=str, default="CNN",
                    choices=["CNN", "CNN_EXTRA", "HI_CNN"],
                    help="The algorithm to use")

parser.add_argument("-n", "--name", type=str, default="no_name",
                    help="A custom name to give the saved log and model files")

args = parser.parse_args()

if args.run and args.name == "no_name":
    parser.error("You should provide a name when running a learning session")

if args.type in ["CNN"] and args.test_episodes == 0:
    parser.error("You should specify the number of testing episodes, other wise the model cannot be evaluated")

# Suppress the many unnecessary TensorFlow warnings
import os, sys
import tensorflow as tf
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto( device_count = {'XLA_CPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Create the simulation
from Simulation.forest_fire import ForestFire
forestfire = ForestFire()

# Start learning straight away
if args.run:
    print(f"\nRunning {args.type} in the "
        f"{args.environment} mode with a size of "
        f"{args.size}*{args.size}.\n")

    # To run the original CNN session
    if args.type in ["CNN"]:
        from CNN import CNN
        Agent = CNN(forestfire, args.name, args.environment, args.size)
        Agent.create_memories(args.train_memories, args.val_memories)               # create data to train and validate
        counter = 0
        n = 30
        while counter < n:
            Agent.train()                                                           # train 30 networks
            Agent.test(args.test_episodes)                                          # test 30 models on 100 new levels
            counter += 1

    # To run the CNN session with more data
    if args.type in ["CNN_EXTRA"]:
        from CNN_EXTRA import CNN_EXTRA
        Agent = CNN_EXTRA(forestfire, args.name, args.environment, args.size)
        Agent.create_memories(args.train_memories, args.val_memories)               # create data to train and validate
        counter = 0
        n = 30
        while counter < n:
            Agent.train()                                                           # train 30 networks
            Agent.test(args.test_episodes)                                          # test 30 models on 100 new levels
            counter += 1

    # To run the Human interactive CNN session with data from levels that previous model failed to successfully finish
    if args.type in ["HI_CNN"]:
        from HI_CNN import HI_CNN
        Agent = HI_CNN(forestfire, args.name, args.environment, args.size)
        Agent.prev_train()                                                          # train a model
        Agent.create_memories(args.train_memories, args.val_memories)               # create data from levels that this model failed to successfully finish
        counter = 0
        n = 30
        while counter < n:
            Agent.train()                                                           # train 30 networks
            Agent.test(args.test_episodes)                                          # test 30 models on 100 new levels
            counter += 1


def get_environment():
    return args.environment


def get_size():
    return args.size
