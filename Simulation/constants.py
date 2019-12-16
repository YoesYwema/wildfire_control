'''
> Simulation constants:

n_actions:
Number of allowed actions. The first 4 actions correspond to N,S,E,W.
The 5th action is toggle dig, if that option is set to True.
Set this number to one higher than the number of actions to allow
for a "do nothing" action.

allow_dig_toggle:
Set the 5th action to toggle between digging and not digging. If disabled,
the 5th action is do nothing, otherwise the 6th action is do nothing.

a_speed:
Number of steps agent can execute before the environment updates: Both equal

debug:
Allow collection of logging info etc: 0 = off, 1 = some, 2 = all

'''

from __main__ import args

A_SPEED = 1
# determines from command line what the size of the area is
SIZE = args.size

AMOUNT_OF_HOUSES = 3 * ((SIZE*SIZE)/100)

# determines from command line what the environment should look like
MAKE_HOUSES = False
MAKE_RIVER = False
if args.environment == "forest_houses":
    MAKE_HOUSES = True
if args.environment == "forest_river":
    MAKE_RIVER = True
if args.environment == "forest_houses_river":
    MAKE_HOUSES = True
    MAKE_RIVER = True

# Metadata and CNN parameters
METADATA = {
    # Simulation constants
    "width" : SIZE,
    "height": SIZE,
    "debug" : 1,
    "n_actions"    : 5,
    "a_speed"      : A_SPEED,
    "a_speed_iter" : A_SPEED,
    "make_houses" : MAKE_HOUSES,
    "make_rivers"  : MAKE_RIVER,
    "wind"  : [0.54, (0, 0)], # random
    "containment_wins" : True,
    "allow_dig_toggle" : True,
    "amount_of_houses" : AMOUNT_OF_HOUSES,

    # Learning rate for the DCNN
    "alpha"          : 0.005
}
