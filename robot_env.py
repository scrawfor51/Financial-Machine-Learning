from TabularQLearner import TabularQLearner
import numpy as np
import sys
import timeit

TRIPS_WITHOUT_DYNA = 500
TRIPS_WITH_DYNA = 50
FAILURE_RATE = 0.2

MOVE_OFFSET = ( (-1,0), (0,1), (1,0), (0,-1) )

def find_unique_location (world, item):
  # Returns a tuple of the x,y coordinate of an item in the world.
  # The item is identified by a one-character string representation
  # as it appears in the world text file, e.g. 'g' for the goal.
  #
  # If the item appears more than once, one is arbitrarily chosen.
  # If the item does not appear, None is returned.

  result = np.where(world == item)
  return (result[0][0], result[1][0]) if len(result[0]) > 0 else None


def calc_state (loc):
  # Quantizes the state to a single number: row*10 + column.
  return loc[0]*10 + loc[1]


def query_world (world, s, a):
  # Apply the action to the state and return the new state and reward.

  # See if the attempted action succeeded.
  if np.random.random() < FAILURE_RATE:
    a = np.random.randint(4)

  # Determine the result of the action.
  s_prime = s[0] + MOVE_OFFSET[a][0], s[1] + MOVE_OFFSET[a][1]
  row, col = s_prime

  # Ensure the action was possible, roll back if needed, and set the reward.
  if row < 0 or row >= world.shape[0] or col < 0 or col >= world.shape[1]:
    # Ran into exterior wall.
    s_prime, r = s, -1
  elif world[s_prime] == 'X':
    # Ran into interior obstacle.
    s_prime, r = s, -1
  elif world[s_prime] == 'q':
    # Walked into quicksand.
    r = -100
  elif world[s_prime] == 'g':
    # Walked into goal.
    r = 1
  else:
    # Legal move to empty space with default reward.
    r = -1

  return s_prime, r


def test_world (world, dyna, trips, show_path = True):
  # Run an experiment with a persistent learning taking many trips through a robot world,
  # possibly with dyna iterations of the Dyna-Q algorithm after each experience,
  # and return the median trip reward.

  # Create a learner to interact with the world.
  eps = 0.98 if dyna == 0 else 0.5
  eps_decay = 0.999 if dyna == 0 else 0.99

  if dyna > 0:
    Q = TabularQLearner(states = world.size, actions = 4, epsilon = eps, epsilon_decay = eps_decay, dyna = dyna)
  else:
    Q = TabularQLearner(states = world.size, actions = 4, epsilon = eps, epsilon_decay = eps_decay)

  # Initialize the robot and goal position.
  start = find_unique_location(world, 's')
  goal = find_unique_location(world, 'g')

  # Remember the total rewards of each trip individually.
  trip_rewards = []

  # Each loop is one complete trip through the maze.
  for i in range(trips):

    # A new trip starts with the robot at the start with no rewards.
    # Get the initial action.
    s = start
    trip_reward = 0
    a = Q.test(calc_state(s))

    # There is a step limit per trip to prevent infinite execution.
    steps_remaining = 10000

    if show_path: path = world.copy()

    # Each loop is one step in the maze.
    while s != goal and steps_remaining > 0:

      # Apply the most recent action and determine its reward.
      s, r = query_world(world, s, a)

      if show_path: path[s] = '+'

      # Allow the learner to experience what happened.
      a = Q.train(calc_state(s), r)

      # Accumulate the total rewards for this trip.
      trip_reward += r

      # Elapse time.
      steps_remaining -= 1

    # Remember the total reward of each trip.
    trip_rewards.append(trip_reward)

    # Print a line of what happened this trip.
    if show_path:
      print (np.array2string(path, separator='', formatter={'str_kind': lambda x: x}))
      found_goal = "     GOAL NOT REACHED" if s != goal else ""
      print (f"Trip {i}, reward: {trip_reward:.04f}{found_goal}")

  return np.median(np.array(trip_rewards))


if __name__ == '__main__':
  # Load the world, instantiate a Q-Learning agent, and help it navigate.

  # The program requires a single command-line argument, which is the path
  # to a world file.  No default is applied.
  #
  # If a second argument is supplied, it is the number of dyna iterations
  # to run after each real experience.  Defaults to zero.
  
  start = timeit.default_timer()
  
  if len(sys.argv) < 2:
    print ("Usage: python robot_env.py [path_to_world_file] [optional: dyna #]")
    exit()
  
  world_file = sys.argv[1]
  world = np.genfromtxt(world_file, dtype=np.dtype('U'), delimiter=1)
  dyna = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  trips = TRIPS_WITH_DYNA if dyna > 0 else TRIPS_WITHOUT_DYNA
  stop = timeit.default_timer()
  
  # Want to see each trip as a visual map?  Use this line.
  score = test_world(world, dyna, trips)

  # Tired of all the printing?  Use this line instead of the above.
  #score = test_world(world, dyna, trips, show_path = False)
  print("Time to run: ", stop-start)
  
  print (f"After {trips} trips, median trip reward was: {score:.4f}")

