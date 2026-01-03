import numpy as np
from env import GridWorld
from agent import SimplePolicy, assign_Targets_Policy, orient_rob, collision_det
from llm_expert import query_llm, query_simulator, calc_targets, assign_targets, calc_moveDir, get_orientation, refine_orientation, obstacle_nav
from visualizer import draw_grid

import torch
from collections import deque
import pygame
import math


GRID_SIZE = 22

env = GridWorld(GRID_SIZE, 2) # inputs: total grid size, padding

# make policies
policy_a = SimplePolicy(input_size=2)
policy_b = SimplePolicy(input_size=2)
assign_targets_policy = assign_Targets_Policy(input_size=18) #Ealrlier it was 12 we made input more expressive
orient_rob = orient_rob(input_size=4) # policy for orientation
collision = collision_det(input_size=6)

# data holders for policies
data_a, data_b = deque(), deque() # to control robots
assign_target_data = deque() # to get targets
orient_data = deque()
collision_data = deque()

print("training")

def change_state(state, grid_size):
    """
    Invert the y-coordinates of the state for Pygame rendering.

    Args:
        state (dict): A dictionary with keys 'robot_a', 'robot_b', 'box', 'goal'.
        grid_size (int): The size of the grid (e.g., 7 for a 7x7 grid).

    Returns:
        dict: New state with inverted y-coordinates.
    """
    def invert_y(pos):
        return (pos[0], grid_size - pos[1])

    new_state = {
        "robot_a": invert_y(state["robot_a"]),
        "robot_b": invert_y(state["robot_b"]),
        "box": invert_y(state["box"]),
        "goal": invert_y(state["goal"]),
    }
    return new_state


def input_for_nn(string):
    # turns rx and tx into a tensor and ry ty into another then returns both in a list
    if string == "robot_a":
        to_return_x = torch.tensor([state[string][0], state['target_a'][0]], dtype=torch.float32)
        to_return_y = torch.tensor([state[string][1], state['target_a'][1]], dtype=torch.float32)
    elif string == "robot_b":
        to_return_x = torch.tensor([state[string][0], state['target_b'][0]], dtype=torch.float32)
        to_return_y = torch.tensor([state[string][1], state['target_b'][1]], dtype=torch.float32)
    else:
        return "Must assign valid robot"
    return [to_return_x, to_return_y] # /20 to normalize for better training


def multi_hot_encode(move_dir):

    # Initialize a 3-value zero vector
    encoding = [float(0)] * 3

    # Encode dx (-1→0, 0→1, 1→2)
    pos = move_dir + 1  # Convert [-1, 0, 1] to [0, 1, 2]
    encoding[pos] = 1.0

    return encoding


def custom_one_hot(labels, basis_classes=None, num_classes=None):
    """
    Args:
        labels: single label or list of labels (int, float, or str)
        basis_classes: list of all classes in desired order (optional)
        num_classes: total number of classes (optional)

    Returns:
        Tensor of shape (N, num_classes) with one-hot encoded labels
    """
    # Step 1: Ensure labels is a list
    if not isinstance(labels, (list, tuple)):
        labels = [labels]

    # Step 2: Deduce basis_classes if not given
    if basis_classes is None:
        basis_classes = sorted(set(labels))  # consistent ordering
    else:
        basis_classes = list(basis_classes)

    # Step 3: Build class-to-index map
    class_to_idx = {cls: i for i, cls in enumerate(basis_classes)}

    # Step 4: Map each label to index (raise error if label not in basis)
    try:
        label_indices = [class_to_idx[lbl] for lbl in labels]
    except KeyError as e:
        raise ValueError(f"Label {e} not in provided basis_classes: {basis_classes}")

    # Step 5: Determine final number of classes
    final_num_classes = num_classes if num_classes is not None else len(basis_classes)

    # Step 6: Generate one-hot tensor
    label_tensor = torch.tensor(label_indices, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=final_num_classes)

    return one_hot.float()

def collides_with(checkForIndex, nextPos, obstacles):

    index = -1
    for i, obs in enumerate(obstacles):
        if i != checkForIndex and obs == nextPos:
            index = i
    return index



def one_hot_label(dir, robot, target):
    dir_x = dir[0]
    dir_y = dir[1]
    rx = robot[0]
    ry = robot[1]
    tx = target[0]
    ty = target[1]

    ### angle theta
    # Compute the denominator r sqrt(x^2 + y^2)
    r = math.sqrt(dir_x ** 2 + dir_y ** 2)
    #Compute the angle in radians
    angle_radians = math.acos(dir_x / (r + 10e-12))
    # angle in degrees
    angle = round( math.degrees(angle_radians) )
    # To get right angles in all quadrants
    if dir_y < 0:
        angle = 360 - angle

    # print("angle: ", angle)

    # Map to closest angle in [0, 45, 90, 135, 180, 225, 270, 315]
    possible_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    #closest_angle = min(possible_angles, key=lambda x: abs(x - angle))

    # Create one-hot encoded vector
    angle_index = possible_angles.index(angle)
    one_hot_vector = torch.zeros(len(possible_angles))
    one_hot_vector[angle_index] = 1.0

    ### which side the target is one
    dx = tx - rx
    dy = ty - ry

    r_target = math.sqrt(dx ** 2 + dy ** 2)
    angle_radians = math.acos(dx / (r_target + 10e-12))
    angle_to_tar = round(math.degrees(angle_radians))
    if dy < 0:
        angle_to_tar = 360 - angle_to_tar

    if angle_to_tar > angle:
        one_hot_vector = torch.cat((one_hot_vector, torch.tensor([1.0])))
    else:
        one_hot_vector = torch.cat((one_hot_vector, torch.tensor([0.0])))


    return one_hot_vector



cnt_true = 0
cnt_total = 0
skip_episode = False  # Flag to skip the current episode

for episode in range(25):  # DAgger-like episodes
    if skip_episode:
        skip_episode = False  # Reset the flag
        continue  # Skip to the next episode

    state = env.reset()
    done = False
    total_reward = 0
    step_counter = 1
    max_steps = 3*GRID_SIZE
    #obstacles = [state['robot_a'], state['robot_b'], state['box'], state['goal']] # create set of obstacles when environment created
    print(episode)

    while not done and step_counter < max_steps:
        obstacles = [state['robot_a'], state['robot_b'], state['box'], state['goal']] + state['obstacles']  # create list of obstacles when environment created

        # Ask expert; options: local, api, simulator ; output is json / dictionary
        #expert_action = query_llm(state, "local")
        #expert_action = query_simulator(state)
        ta, tb = calc_targets(state['box'], state['goal'])
        s = assign_targets(state, ta, tb) # returns true if need swapping
        tax, tay = ta[0], ta[1] # unswitched targets pulled for training input before they are switched
        tbx, tby = tb[0], tb[1]
        if s: # switch if wrongly assigned by default
            #print("Switched while training")
            ta, tb = tb, ta
            cnt_true += 1
        cnt_total += 1
        # Now we are ready to compute direction steps for each robot
        aDir, bDir = calc_moveDir(state['robot_a'], state['robot_b'], ta, tb)


        ### orientation
        orient_a, orient_b = get_orientation(ta, tb, state['box']) # orientation
        orient_a_r, orient_b_r = refine_orientation(orient_a, orient_b) # refine orientation to make it look more realistic

        state['switch'] = s # whether targets were switched
        state['target_a'] = (ta[0], ta[1])
        state['target_b'] = (tb[0], tb[1])
        #print("Method IsSwitch in trian data: ", state['switch'])
        # update orientations of refined angles in state
        state["robot_a_orient"] = orient_a_r
        state["robot_b_orient"] = orient_b_r

        # ///////////////////////////////////////////////////////////////////////
        # //////////////////// obstacle avoidance based on directios to move
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////
        flgA = 1
        flgB = 1
        # only check for obstacles if robot is not already on target
        #### robot A
        rob_a_next = (state['robot_a'][0] + aDir[0],
                      state['robot_a'][1] + aDir[1])  # get the next location of both robots

        collision_detected_a = False
        collision_detected_b = False
        #if rob_a_next != state['target_a']:
        ##### input for nn
        #if rob_b_next != state['target_b']:
        index = collides_with(0, rob_a_next, obstacles) # robot _a collides with index if index > -1
        if index > -1:
            collision_detected_a = True # to make input for nn
            if index == 0 or index == 1:
                print("Robo-Collision: A collides with -->", index)
                if state['robot_b'] == state['target_b']:
                    aDir = obstacle_nav(state['robot_a'], aDir, state['target_a'], obstacles)
                    next = (state['robot_a'][0] + aDir[0],
                            state['robot_a'][1] + aDir[1])  # update this robot in obstacles as soon as dir recieved
                    obstacles[0] = next
                else:
                    aDir = (0, 0) # if other robot not on target, this one waits for it to pass
            else:
                flgA = 0
                aDir = obstacle_nav(state['robot_a'], aDir, state['target_a'], obstacles)
                if index >3 and state['target_a'] == obstacles[index]:
                    print("dir a", aDir)
                    state['target_a'] = state['robot_a']
                    ta = state['robot_a']
                    aDir = (0,0)
                next = (state['robot_a'][0] + aDir[0],
                        state['robot_a'][1] + aDir[1])  # update obstacles as soon as dir recieved
                obstacles[0] = next  # update obstacles as soon as dir recieved
                #print("dir a", aDir)

        # robot B
        rob_b_next = (state['robot_b'][0] + bDir[0], state['robot_b'][1] + bDir[1])
        index = collides_with(1, rob_b_next, obstacles)  # robot _a collides with index if index > -1
        if index > -1:
            collision_detected_b = True # to make input to nn
            if index == 0 or index == 1:
                print("Robo-Collision: B collides with -->", index)
                if state['robot_a'] == state['target_a']:
                    bDir = obstacle_nav(state['robot_b'], bDir, state['target_b'], obstacles)
                    next = (state['robot_b'][0] + bDir[0],
                            state['robot_b'][1] + bDir[1])  # update obstacles as soon as dir recieved
                    obstacles[1] = next
                else:
                    bDir = (0, 0)
            else:
                # flgB = 0
                #print("dir_b: ", bDir)
                bDir = obstacle_nav(state['robot_b'], bDir, state['target_b'], obstacles)
                if index >3 and state['target_b'] == obstacles[index]:
                    print("dir b: ", bDir)
                    state['target_b'] = state['robot_b']
                    tb = state['robot_b']
                    bDir = (0,0)
                next = (state['robot_b'][0] + bDir[0],
                        state['robot_b'][1] + bDir[1])  # update obstacles as soon as dir recieved
                obstacles[1] = next
                #print("dir_b: ", bDir)
        # ////////////////////////////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////////////////////////////////////

        # print("a", state["robot_a_orient"])
        # print("b", state["robot_b_orient"])
        action = { # to be passed to step update method of environmnet
            "robot_a": aDir, # action for robot a
            "robot_b": bDir,
            "target_a": ta,
            "target_b": tb,
            "robot_a_orient": orient_a_r, # send in the refined angles
            "robot_b_orient": orient_b_r,
        }


        #################### Collision detection
        ####################
        if collision_detected_a == True:
            #(robot, target, direction) > theta (0-7) and if target is left or right (+ve, -ve)
            # output : total 9 neurons

            # Flatten obstacle coordinates without modifying the original list
            flat_obstacles = [coord for obj in obstacles for coord in obj]

            # input 1D tensor (robot, target, direction)
            collision_input = torch.tensor([
                state['robot_a'][0], state['robot_a'][1],
                aDir[0], aDir[1],
                state['target_a'][0], state['target_a'][1]
            ], dtype=torch.float32)
            #print(collision_input)

            collision_label = one_hot_label(aDir,  state['robot_a'], state['target_a'])
            #print("final label:", collision_label)

            # vec_a = multi_hot_encode(aDir[0]) # use already defiend method
            # vec_b = multi_hot_encode(aDir[1])
            # label = vec_a + vec_b
            # collision_label = torch.tensor([label], dtype=torch.float32)

            collision_data.append((collision_input, collision_label)) # append data to list/deque


        if collision_detected_b == True:
            # Flatten obstacle coordinates without modifying the original list
            flat_obstacles = [coord for obj in obstacles for coord in obj]

            # input 1D tensor
            collision_input = torch.tensor([
                state['robot_b'][0], state['robot_b'][1],
                bDir[0], bDir[1],
                state['target_b'][0], state['target_b'][1]
            ], dtype=torch.float32)
            #print(collision_input)

            collision_label = one_hot_label(bDir,  state['robot_b'], state['target_b'])
            #print("final label:", collision_label)

            # vec_a = multi_hot_encode(bDir[0])  # use already defined method
            # vec_b = multi_hot_encode(bDir[1])
            # label = vec_a + vec_b
            # collision_label = torch.tensor([label], dtype=torch.float32)

            collision_data.append((collision_input, collision_label)) # append data to list/deque


        ###########################
        ###################### orientation
        # input for orientation
        #in_orient = torch.tensor([ta[0], ta[1], tb[0], tb[1], state["box"][0], state["box"][1]], dtype=torch.float32)
        in_orient_a = torch.tensor([ta[0], ta[1], state["box"][0], state["box"][1]], dtype=torch.float32)
        in_orient_b = torch.tensor([tb[0], tb[1], state["box"][0], state["box"][1]], dtype=torch.float32)
        #print(in_orient)

        # label for orientation:

        # update orientations in state
        # state["robot_a_orient"] = orient_a
        # state["robot_b_orient"] = orient_b

        orient_label_a = custom_one_hot(orient_a, basis_classes=[0, 90, 180, 270])
        orient_label_b = custom_one_hot(orient_b, basis_classes=[0, 90, 180, 270])
        # print(orient_label_a)
        # print(orient_a)

        # Concatenate to get multi hot encoded vector with both target orientation included
        #concatenated_orient_label = torch.cat([orient_label_a, orient_label_b], dim=0).flatten()
        orient_data.append((in_orient_a, orient_label_a))
        orient_data.append((in_orient_b, orient_label_b))
        # print(concatenated_orient)
        #########################################
        ######################## Make data for target NN // tensor with box then goal
        ##################################################################################
        # input
        bx, by = state['box']
        gx, gy = state['goal']
        rax, ray = state['robot_a']
        rbx, rby = state['robot_b']
        #s = state["switch"] # label
        # Prep input and labels for training Assign_NN_policy
        input_target_nn = torch.tensor([bx, by, gx, gy, rax, ray, tax, tay, rbx, rby, tbx, tby, bx-gx, by-gy, tax-rax, tay-ray, tbx-rbx, tby-rby], dtype=torch.float32)
        #print( "bx-gx, by-gy, gx, gy, rax, ray, rbx, rby, tax, tay, tbx, tby: \n", bx, by, gx, gy, rax, ray, rbx, rby, tax, tay, tbx, tby)
        #### Labels for target NN: switched or not (1 if switched else 0)
        if s > 0.5:
            s = 1
            #softmax_label = [0, 1]
        else:
            s = 0 # sigmoid requires labels between 0 and 1
            #softmax_label = [1, 0]

        #label_target_nn = torch.tensor([softmax_label[0], softmax_label[1]], dtype=torch.float32) # convert label to tensor
        label_target_nn = torch.tensor(s, dtype=torch.float32) # convert label to tensor

        # print("input shape",input_target_nn.shape)
        # print("label shape", label_target_nn.shape)
        # print("input", input_target_nn)
        # print("label", label_target_nn)
        assign_target_data.append((input_target_nn, label_target_nn)) # appended to deque as a tuple

        ########################################Below relevant for training MoveStep NN
        ################################################################ Else Ignore
        ############################curerntly it is not trained every time but saved
        # preparing multi hot labels / outputs for moving NN
        # by converting action / move_step i.e dx = (-1, 0, 1) & dy = (-1,0,1) to 3 output labels of nn [0 0 1]
        # where three labels encode step dx in the same order in negative x, non and in positive x.
        label_a_x = multi_hot_encode(action['robot_a'][0])
        label_a_y = multi_hot_encode(action['robot_a'][1])
        label_b_x = multi_hot_encode(action['robot_b'][0])
        label_b_y = multi_hot_encode(action['robot_b'][1])
        #label_a = multi_hot_encode(expert_action["robot_a"])
        #label_b = multi_hot_encode(expert_action["robot_b"])
        #For move action step 1 get updated target and rob values to pass tehm to Move_NN_policy
        # location of robot + target for each robot, x and y separately
        input_a_x, input_a_y = input_for_nn("robot_a") #input_for_nn is a method to get updated inputs from state
        input_b_x, input_b_y = input_for_nn("robot_b") #
        # print("input_a_x", input_a_x)
        # print("input_a_y", input_a_y)
        # print("label_a_x", label_a_x)
        # print("label_a_y", label_a_y)
        # append state-action pairs, used in training the agent
        data_a.append((input_a_x, label_a_x)) # moveStep policy trainijng data  for robo a
        data_a.append((input_a_y, label_a_y))
        data_b.append((input_b_x, label_b_x)) # for robo b
        data_b.append((input_b_y, label_b_y))
        ############################################### MoveStep TRainijng data prepped and appended above
        #########################################################################################################

        ## Now time to update environment or tke actions generated so far
        # Step the environment , update the environment
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state


        # draw to visualize
        #draw_grid(state, GRID_SIZE)
        if draw_grid(state, GRID_SIZE):
            skip_episode = True
            break  # Skip to the next episode
        pygame.time.delay(500) # slow down movements to see each step

        max_steps += 1
    ################################################################Trainig Various NN_policues
    #######################################################################
    # Trains moveStep predictiojn policies for each root after each episode
    # Now no needd to traijn every time models cab saved and loaded
    policy_a.train(policy_a, data_a)
    policy_b.train(policy_b, data_b)

    # Traijn assign target NN policy
    # assign_targets_policy.train(data=assign_target_data) # Train target policy with target data

    # train orientation policy
    #orient_rob.train(data=orient_data)

    # train collision
    #collision.train(data=collision_data)
    #print("num examples in collision data: ", len(collision_data))

# assign_targets_policy.plot_training_metrics()

# Training complete can be saved below before movign to test
print("Training complete. Trained Models can be saved now....")
print("True / Total: ", cnt_true / cnt_total)

# Save the models:  state_dict weights of the model
#torch.save(policy_a.state_dict(), 'models/robot_a_w.pth')
#torch.save(policy_b.state_dict(), 'models/robot_b_w.pth')

torch.save(collision.state_dict(), 'models/current_collision_weights.pth')
torch.save(collision, 'models/current_collision_full_model.pth')

# plot here, after model is saved
# assign_targets_policy.plot_training_metrics()
#orient_rob.plot_training_metrics()
#collision.plot_training_metrics()
policy_a.plot_training_metrics()

# Later, to load:
#policy_a = SimplePolicy(input_size=2)  # Recreate the model structure
#policy_b = SimplePolicy(input_size=2)

# reload  model weights from files
policy_a.load_state_dict(torch.load('models/robot_a_w.pth'))
policy_b.load_state_dict(torch.load('models/robot_b_w.pth'))
assign_targets_policy.load_state_dict(torch.load('models/target_assign_w_single_out_neuron_sigmoid0.pth'))
orient_rob.load_state_dict(torch.load('models/orientation_weights_adam1.pth'))

print("Train model saved. Now Testing....")
# send in policies and testing data
# Testing data will contain the robot position and target position
# Moves predicted will be visualized, but should ideally be compared to the simulator's

#############################################################
## Testing module

def indices_to_robot_control(index):
    """
    Converts index to movement
    """
    return index - 1 # all indices will be from 0, 1, 2, su subtractign 1 gives -1, 0, 1

def get_moveDir_from_NN():
    ############################# Below inputs to NN_movestep use the state to get action
    input_a_x, input_a_y = input_for_nn("robot_a")  # input_for_nn is a method to get updated inputs from state
    input_b_x, input_b_y = input_for_nn("robot_b")

    # get predicted move-actions from agents (0, 1, 2). Twice for each robot (x and y)
    a_index_x = predict_action(policy_a, input_a_x)
    a_index_y = predict_action(policy_a, input_a_y)

    b_index_x = predict_action(policy_b, input_b_x)
    b_index_y = predict_action(policy_b, input_b_y)

    # b_index_x, b_index_y = predict_action(policy_b, input_a_y)
    # convert prediction (0,1, 2) to move-directions (-1,0,1): dir = pred - 1
    a_action_x = indices_to_robot_control(a_index_x)  # robot a
    a_action_y = indices_to_robot_control(a_index_y)
    b_action_x = indices_to_robot_control(b_index_x)  # robot b
    b_action_y = indices_to_robot_control(b_index_y)

    # prepare dictionary to update environment, giving directions for robots and targets
    action = {
        "robot_a": [a_action_x, a_action_y],
        "robot_b": [b_action_x, b_action_y],
        "target_a": [tax, tay],
        # They must be now extracted from state as state has the most updated adn swapped targets
        "target_b": [tbx, tby],

        # To use targets coming in from the expert:
        # "target_a": [expert_action['target_a'][0], expert_action['target_a'][1]],
        # "target_b": [expert_action['target_b'][0], expert_action['target_b'][1]],
    }


#input("Press Enter to start...")

def one_hot_to_angle(one_hot_tensor):
    """
    Converts a 4-value one-hot encoded tensor to its corresponding angle.
    """
    # Define the angle mapping
    angle_map = [0, 90, 180, 270]

    # Convert one-hot to angle
    angle = torch.argmax(one_hot_tensor, dim=-1)  # torch.argmax returns index of largest value in tensor
    return angle_map[angle]  # Map to angle



total_cnt = 0
wrong_cnt = 0
falseTrue_cnt = 0

total_cnt_orient = 0
wrong_orient_cnt = 0
falseTrue_cnt_orient = 0

for episode in range(10):  # choose any number of test episodes
    state = env.reset()
    done = False
    total_reward = 0
    step_counter = 1
    max_steps = 3*GRID_SIZE

    print(f"\n=== Test Episode {episode + 1} ===")

    while not done and step_counter < max_steps:

        # call expert to get targets and also correct labels for test accuracy
        # expert_action = query_llm(state, "local")
        #expert_action = query_simulator(state)
        ta, tb = calc_targets(state['box'], state['goal'])
        s_e = assign_targets(state, ta, tb)  # returns true if need swapping

        ######################## Make data for target NN // tensor with box then goal
        ##################################################################################
        # input
        bx, by = state['box']
        gx, gy = state['goal']
        rax, ray = state['robot_a']
        rbx, rby = state['robot_b']
        #s = state["switch"]  # label not required as while tsting it will come from policy
        tax, tay = ta[0], ta[1]  # unswitched targets pulled before they are switched for trainig
        tbx, tby = tb[0], tb[1]
        # Prep input and labels for training Assign_NN_policy
        input_target_nn = torch.tensor(
            [bx, by, gx, gy, rax, ray, tax, tay, rbx, rby, tbx, tby, bx - gx, by - gy, tax - rax, tay - ray, tbx - rbx,
             tby - rby], dtype=torch.float32)
        s = assign_targets_policy.predict(input_target_nn)
        # print('output shape: ', s.shape)

        raw_s = s

        #print("IsSwitch from model NN: ", s)
        if s > 0.5: s = 1 # with sigmoid, output will always be between 0 and 1
        else: s = 0

        # for softmax, there are 2 outputs
        # if s[0] < s[1]:
        #     s = 1 # with sigmoid, output will always be between 0 and 1
        # else:
        #     s = 0
        #print(" NN rounded off IsSwitch: ", s)
        #print ("Simulator isSwich: ", s_e)

        if s != s_e:
            # print('////////Error s and s_e do not match/////////////////')
            # print("Raw s from model NN: ", raw_s)
            # print("IsSwitch s from model NN: ", s)
            # print("Simulator isSwitch: ", s_e)
            wrong_cnt += 1
            if s_e: falseTrue_cnt += 1
        total_cnt += 1

        if s:  # switch if wrongly assigned by default
            ta, tb = tb, ta
        # Now we are ready to compute direction steps for each robt
        state['switch'] = s  # whether targets were switched
        state['target_a'] = (ta[0], ta[1])
        state['target_b'] = (tb[0], tb[1])
        aDir, bDir = calc_moveDir(state['robot_a'], state['robot_b'], ta, tb)

        ##################################### obstacle detection
        ######################################################

        obstacles = [state['robot_a'], state['robot_b'], state['box'], state['goal']] + state['obstacles']
        rob_a_next = (state['robot_a'][0] + aDir[0],
                      state['robot_a'][1] + aDir[1])  # get the next location of robot a
        
        index = collides_with(0, rob_a_next, obstacles) # robot _a collides with index if index > -1
        if index > -1:
            aDir = collision.obstacle_nav_nn(state['robot_a'], aDir, state['target_a'], obstacles)

        rob_b_next = (state['robot_b'][0] + bDir[0],
                      state['robot_b'][1] + bDir[1])  # get the next location of robot a
        index = collides_with(1, rob_b_next, obstacles) # robot _a collides with index if index > -1
        if index > -1:
            bDir = collision.obstacle_nav_nn(state['robot_b'], bDir, state['target_b'], obstacles)
            #print("new dir for robot b:", bDir)

        #####################################


        ############## orientation
        orient_a, orient_b = get_orientation(ta, tb, state['box'])  # orientation

        # input data for orientation
        # in_orient = torch.tensor([ta[0], ta[1], tb[0], tb[1], state["box"][0], state["box"][1]], dtype=torch.float32)
        in_orient_a = torch.tensor([ta[0], ta[1], state["box"][0], state["box"][1]], dtype=torch.float32)
        in_orient_b = torch.tensor([tb[0], tb[1], state["box"][0], state["box"][1]], dtype=torch.float32)
        # get pred from nn
        orient_pred_a = orient_rob.predict(in_orient_a)
        orient_pred_b = orient_rob.predict(in_orient_b)
        # round the recieved answers so they are 0 or 1
        rounded_orient_pred_a = torch.round(orient_pred_a)
        rounded_orient_pred_b = torch.round(orient_pred_b)
        # print(rounded_orient_pred)
        # orient_a = rounded_orient_pred[:4]
        # orient_b = rounded_orient_pred[4:]

        angle_a = one_hot_to_angle(rounded_orient_pred_a)
        angle_b = one_hot_to_angle(rounded_orient_pred_b)

        total_cnt_orient += 1

        if angle_a != orient_a and angle_b != orient_b:
            wrong_orient_cnt +=1


        orient_a_r, orient_b_r = refine_orientation(angle_a, angle_b) # refine orientation to make it look more realistic
        #orient_a_r, orient_b_r = refine_orientation(orient_a, orient_b)

        #########################

        action = {  # to be passed to step update method of environment
            "robot_a": aDir,  # action for robot a
            "robot_b": bDir,
            "target_a": ta,
            "target_b": tb,
            "robot_a_orient": orient_a_r,
            "robot_b_orient": orient_b_r,
        }

        # Update environment with predicted actions // its here target is also updated
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        # Visualize the step or state env after taking step
        #draw_grid(state, GRID_SIZE)
        if draw_grid(state, GRID_SIZE):
            skip_episode = True
            break  # Skip to the next episode
        pygame.time.delay(75)

        max_steps += 1

    print(f"Total reward in test episode {episode + 1}: {total_reward}")

print('\n///////////////////////////////////////////////////////////////////// Assignment policy test results:')
print("True / Total in training data: ", cnt_true / cnt_total)
print(" Total Wrong: ", wrong_cnt, " out of Total: ", total_cnt)
print("Total Wrong fraction: ", wrong_cnt / total_cnt)
print("False Positives: ", falseTrue_cnt, "  False negatives : ", wrong_cnt - falseTrue_cnt)
print("FalsePositives Fraction: ", falseTrue_cnt/wrong_cnt, "  FalseNegatives Fraction: ", 1 - falseTrue_cnt/wrong_cnt)

print('\n Orientation Policy Test Results: ')
print("True / Total in training data: ", cnt_true / cnt_total)
print(" Total Wrong: ", wrong_orient_cnt, " out of Total: ", total_cnt_orient)
print("Total Wrong fraction: ", wrong_orient_cnt / total_cnt_orient)
# print("False Positives: ", falseTrue_cnt, "  False negatives : ", wrong_cnt - falseTrue_cnt)
# print("FalsePositives Fraction: ", falseTrue_cnt/wrong_cnt, "  FalseNegatives Fraction: ", 1 - falseTrue_cnt/wrong_cnt)


# Without sigmoid (MSE) single output
# 120 episodes
# True / Total in training data:  0.5340143951706524
# Total Wrong:  87  out of Total:  1158
# Total Wrong fraction:  0.07512953367875648
# False Positives:  41   False negatives :  46
# FalsePositives Fraction:  0.47126436781609193   FalseNegatives Fraction:  0.5287356321839081


# With sigmoid/BCE single output
# 120 episodes
# True / Total in training data: 0.48368382886149386
# Total Wrong:  75  out of Total:  1059
# Total Wrong fraction:  0.0708215297450425
# False Positives:  33   False negatives :  42
# FalsePositives Fraction:  0.44   FalseNegatives Fraction:  0.56

# Softmax
# 120 episodes
# True / Total in training data:  0.4879002688829137
#  Total Wrong:  84  out of Total:  1054
# Total Wrong fraction:  0.07969639468690702
# False Positives:  61   False negatives :  23
# FalsePositives Fraction:  0.7261904761904762   FalseNegatives Fraction:  0.27380952380952384