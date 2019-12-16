import pickle
import numpy as np
import random, time, json, os

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint



# Makes a function 'getch()' which gets a char from user without waiting for enter
try:
    # Windows
    from msvcrt import getch
except ImportError:
    # UNIX
    def getch():
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Extract memory from folder
def extract_from_memory(path):
    with open(path + random.choice(os.listdir(path)), "rb") as pf:
        (state, action, done) = pickle.load(pf)
        pf.close()
    return state, action, done


class CNN:
    def __init__(self, sim, name="no_name", environment="forest", size=10, verbose=True):
        # Constants and such
        self.sim = sim
        self.name = name
        self.METADATA = sim.METADATA
        self.action_size = self.sim.n_actions
        self.DEBUG = sim.DEBUG
        self.verbose = verbose
        self.environment = environment
        self.size = size

        # Information to save to file
        self.logs = {
            'amount_of_fires_contained' : 0,
        }

        # CNN Parameter alpha (learning rate)
        self.alpha = self.METADATA['alpha']  # learning rate

        # Creating the CNN with function .make_network()
        self.model = self.make_network()

        # Print Constants
        if self.verbose:
            width, height = self.METADATA['width'], self.METADATA['height']
            print("\n\t[Parameters]")
            print("[size]", f"{width}x{height}")

    # Performing runs in which a Human has to provide training input (delivering data)
    def create_memories(self, train_memories, validation_memories):
        # Loops over amount of episodes which is specified in input arguments
        for episode in range(train_memories + validation_memories):
            # Print some information about the episode
            print(f"[Episode {episode + 1}]")
            # Initialize the done flag, in order that the system knows when the episode is finished
            done = False
            # Initialize the state, and reshape because Keras expects the first dimension to be the batch size
            state = self.sim.reset()
            state = np.reshape(state, [1] + list(state.shape))
            # Maps keyboard to number for a certain action
            key_map = {'w': 0, 's': 1, 'd': 2, 'a': 3, ' ': 4, 'n': 5}

            while not done:
                self.sim.render()
                print("W A S D to move, Space to dig,")
                print("'n' to wait, 'q' to quit.\n")
                # Get action from keyboard
                char = getch()
                if char == 'q':
                    return "Cancelled"
                elif self is not None and len(self.sim.W.border_points) == 0:
                    done = True
                elif char in key_map:
                    action = key_map[char]
                    # Do action, observe environment
                    sprime, score, done,_ = self.sim.step(action)
                    sprime = np.reshape(sprime, [1] + list(sprime.shape))
                    # Store experience in memory if agent is not dead and the performed action is not waiting!
                    if not action == 5 and not done:
                        if episode < train_memories:
                            self.remember(state, action, True, done)
                        else:
                            self.remember(state, action, False, done)
                    # Current state is now next state
                    state = sprime
                else:
                    print("Invalid action, not good for collecting memories.\n")

            # Store the observed experience for last move before episode is done
            if not action == 5 and not done:
                if episode < train_memories:
                    self.remember(state, action, True, done)
                else:
                    self.remember(state, action, False, done)



    # Train the Keras CNN model with samples taken from the memory
    def train(self):
        environment = str(self.environment) + "/"
        size = "size" + str(self.size) + "/"
        # Train variables
        train_path = "train/" + environment + size
        train_states_batch = list()
        train_action_batch = list()
        train_batch = list()
        # Validation variables
        validation_path = "validate/" + environment + size
        validation_states_batch = list()
        validation_action_batch = list()
        validation_batch = list()

        # Amount of available memories for current environment and size
        amount_of_memories = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])
        print("\n[train_memories]", amount_of_memories)
        validation_memories = len([name for name in os.listdir(validation_path) if os.path.isfile(os.path.join(validation_path, name))])
        print("[validation_memories]", validation_memories)

        i = 0
        # Extract train memories
        while i < amount_of_memories:
            x = extract_from_memory(train_path)
            train_batch.append(x)
            i = i + 1

        i = 0
        # Extract validation memories
        while i < validation_memories:
            x = extract_from_memory(validation_path)
            validation_batch.append(x)
            i = i + 1

        # Store the states and the corresponding action for this batch
        for state, action, done in train_batch:
            train_states_batch.append(state[0])
            output = to_categorical(action, num_classes=5, dtype='float32')
            train_action_batch.append(output)

        for state, action, done in validation_batch:
            validation_states_batch.append(state[0])
            output = to_categorical(action, num_classes=5, dtype='float32')
            validation_action_batch.append(output)

        # Convert the batch into numpy arrays for Keras and fit the model
        data_input = np.array(train_states_batch)
        data_output = np.array(train_action_batch)
        val_data_input = np.array(validation_states_batch)
        val_data_output = np.array(validation_action_batch)

        # To prevent overfitting and choose best model according to data on which is only evaluated
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        mc = ModelCheckpoint('best_model', monitor='val_accuracy', mode='max', save_best_only=True)

        # Final command to train model
        self.model.fit(data_input, data_output, validation_data=(val_data_input, val_data_output), epochs=1000, callbacks=[es, mc], verbose=False)

    '''
    Test the behavior learned from training on completely new situations
    without supervision and see whether agent will isolate the fire.
    '''
    def test(self, n_rounds):
        # Variables to keep track of deaths, contained fires and average percentage of healthy cells
        amount_of_deaths = 0
        amount_of_fires_contained = 0
        test_scores = list()
        # Start n episodes to see how the model behaves
        for episode in range(n_rounds):
            # Initialize the done flag, in order that the system knows when the episode is finished
            done = False
            score = 0
            # Initialize the state, and reshape because Keras expects the first dimension to be the batch size
            state = self.sim.reset()
            state = np.reshape(state, [1] + list(state.shape))
            while not done:
                softmax_char = self.model.predict(state)[0]
                char = np.argmax(softmax_char)
                action = char
                # Do action, observe environment
                sprime, score, done, _ = self.sim.step(action)
                sprime = np.reshape(sprime, [1] + list(sprime.shape))
                # Current state is now next state
                state = sprime

            # Keep track of agent deaths
            if len(self.sim.W.agents) == 0:
                amount_of_deaths = amount_of_deaths + 1

            # Keep track of sucesfull(means fire is isolated) episodes
            if self.sim.W.FIRE_ISOLATED == True:
                amount_of_fires_contained += 1
            test_scores.append(score)

            # # Print some information about the episode
            # self.sim.render()
            # print(f"[Episode {episode + 1}]")
            # print(f"\t\tAgent dead: {len(self.sim.W.agents) == 0}")
            # print(f"\t\tFires contained: {amount_of_fires_contained}")
            # print(f"Score: {score}")

        print(f"Amount of times the agent isolated the fire : {amount_of_fires_contained} times")
        self.logs['amount_of_fires_contained'] = amount_of_fires_contained
        self.write_logs()

    # Store an experience in /train/environment/sizeX.
    def remember(self, state, action, train, done):
        # path towards folders in which state + action must be stored
        path = str(self.environment) + "/" + "size" + str(self.sim.W.WIDTH) + "/"
        # create folder names
        name = self.sim.get_name(self.sim.W.WIDTH, str(self.environment))

        # TRAIN DATA
        if train:
            # makes sure correct folder is used for storing examples
            if not os.path.exists("train/" + path):
                os.makedirs("train/" + path)
            # add counter to the end of the name of files
            counter = 0
            while os.path.isfile("train/" + path + name):
                if counter > 0:
                    n_digits_to_delete = len(str(counter))
                    name = name[:-n_digits_to_delete]
                name = name + str(counter)
                counter += 1
            # store object with current state, action and whether agent has died in a file
            with open("train/" + path + name, "wb") as pf:
                pickle.dump((state, action, done), pf)
                pf.close()

        # VALIDATION DATA
        if not train:
            if not os.path.exists("validate/" + path):
                os.makedirs("validate/" + path)
            # add counter to the end of the name of files
            counter = 0
            while os.path.isfile("validate/" + path + name):
                if counter > 0:
                    n_digits_to_delete = len(str(counter))
                    name = name[:-n_digits_to_delete]
                name = name + str(counter)
                counter += 1
            with open("validate/" + path + name, "wb") as pf:
                pickle.dump((state, action, done), pf)
                pf.close()

    # Create the Deep Convolutional Neural Network
    def make_network(self):
        input_shape = (self.sim.W.WIDTH, self.sim.W.HEIGHT, self.sim.W.DEPTH)
        layers = [
            Conv2D(100, kernel_size=2, activation='relu', input_shape=input_shape),

            MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid', data_format=None),

            Conv2D(50, kernel_size=2, activation='relu'),

            Conv2D(25, kernel_size=2,  activation='relu'),

            Flatten(),

            Dense(units=self.action_size, activation='softmax'),
        ]

        model = Sequential(layers)
        # Compile model with categorical crossentropy error loss, the Adam optimizer
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      # And an Adam optimizer with gradient clipping
                       optimizer=Adam(lr=self.alpha, clipvalue=1))
        if self.verbose:
            model.summary()
        return model

    def write_logs(self):
        # Get name for Log file
        name = self.sim.get_name(self.sim.W.WIDTH, self.environment)

        # If the folder doesn't exist, create it
        if not os.path.exists("Logs/"):
            os.makedirs("Logs/")
        # Write all logs into one file for each sort of simulation
        with open("Logs/" + name, 'ab+') as f:
            f.seek(0, 2)  # Go to the end of file
            if f.tell() == 0:  # Check if file is empty
                f.write(json.dumps(self.logs).encode())  # If empty, write an array
            else:
                f.seek(-1, 2)
                # f.truncate()  # Remove the last character, open the array
                f.write('\n'.encode())  # Write the separator
                f.write(json.dumps(self.logs).encode())  # Dump the dictionary
                # f.write(']'.encode())  # Close the array
