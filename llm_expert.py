import numpy as np
import requests
import json
import openai 
from openai import OpenAI
import math

def query_llm(state, choice):
    prompt = f"""
You are controlling two robots A and B, to follow their targets in a grid world.
You are provided with the Current states or positions of two robots and their targets.
- Robot A: {state['robot_a']}
- Robot B: {state['robot_b']}
- target_a: {state['target_a']}
- target_b: {state['target_b']}

Using the above current state you have Generate action for each robot in the following JSON format

{{
  "robot_a": [-1/0/1, -1/0/1],
  "robot_b": [-1/0/1, -1/0/1]
}}

Do not generate any text other than above json format. Do not guess the action randomly. 
You have to compute like an expert the next move direction for each robot so that robots come closer to the targets. A hint is provided below. This is just a hint, you need to apply logic.

hint: move direction for a robot sign [max  [ ( target_X - robot_X), target_Y - robot_Y ) ]]
"""
    
    prompt_1 = f"""
You are controlling two robots A and B, to follow their targets in a grid world.
You are provided with the Current states or positions of two robots and the box state.
- Robot A: {state['robot_a']}
- Robot B: {state['robot_b']}
- box: {state['box']}

Using the above states you have to orient each robot towards the box. Reply in the following JSON format

{{
  "robot_a": [1/2/3/4],
  "robot_b": [1/2/3/4]
}}

Do not generate any text other than above json format. Do not guess the action randomly. 
You have to compute like an expert the next move direction for each robot so that robots come closer to the targets. You need to apply logic.

"""

    if choice == "local":
        print(prompt)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",  # or mistral, etc.
                "prompt": prompt,
                "stream": False
            }
        )

        #print("Ollama raw response:", response.status_code, response.text)
        text = response.json()["response"]
        print(text)

        # Attempt to extract JSON action dictionary from LLM output
        try:
            # In case the model returns extra text, extract the JSON part
            start = text.index('{')
            end = text.index('}', start) + 1
            json_part = text[start:end]
            actions = json.loads(json_part)
            return actions
        except Exception as e:
            print("Failed to parse LLM response:", text)
            raise e

    elif choice == "api":
        api_key = "YOUR_OPENAI_API_KEY_HERE"
        client = OpenAI(api_key=api_key)

        # Use ChatGPT (GPT-4 or GPT-3.5-turbo)
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content
        print(text)

        try:
            # In case the model returns extra text, extract the JSON part
            start = text.index('{')
            end = text.index('}', start) + 1
            json_part = text[start:end]
            actions = json.loads(json_part)
            return actions
        except Exception as e:
            print("Failed to parse LLM response:", text)
            raise e
    return None
#///////////////////////////////LLM asExpert Ends Here /////////////////////////


#//// Self modeled Simulator Expert

def calc_targets(box_pos, goal_pos, isSingleF = False):
    bx, by = box_pos
    gx, gy = goal_pos

    # SINGLE / SAME TARGET POINT FOR BOTH  rtx adn rty are the targets of the robot usually one step behind bo
    if isSingleF:
        rtx = bx + np.sign(bx - gx)
        rty = by + np.sign(by - gy)

        tax = rtx
        tay = rty  # robo A
        tbx = rtx
        tby = rty  # robo B

    # SEPARATE TARGET POINTS FOR BOTH ROBOTS
    else:
        # Case 1: Horizontal Case: y are equal, only diff in x
        if by - gy == 0:

            tbx = bx + np.sign(bx - gx)  # robo As target coordinate computed
            tby = by + 0.5
            tax = bx + np.sign(bx - gx)  # robo B target coordinate computed
            tay = by - 0.5
        # Case 2: Vertical Case x are equal, only diff in y
        elif bx - gx == 0:

            tax = bx + 0.5
            tay = by + np.sign(by - gy)

            tbx = bx - 0.5
            tby = by + np.sign(by - gy)
        # Case 3: 45 degree: neither are equal, box needs to move in both axis to get to goal.
        else:
            tax = bx + np.sign(bx - gx)
            tay = by
            tbx = bx
            tby = by + np.sign(by - gy)

    return [tax, tay], [tbx, tby]


def assign_targets(state, a_tar, b_tar):
    switch = False # decides if need to swap targets and returns switch
    rax, ray = state["robot_a"]
    rbx, rby = state["robot_b"]

    bx, by = state["box"]
    gx, gy = state["goal"]
    # ////////////////////////Assignment of tragets to respective Robots
    # if state["noResetF"] == 0:
    # # Findijng LINE joining G and B : y = mx + b
    if bx != gx:
        eps = 1e-8
        m = (by - gy) / (bx - gx + eps)  # m is +ve if box is on right and top relative to Goal
        b = by - m * bx  # y = mx + b
    else:
        # m = float('inf')
        b = None  # or handle vertical line case explicitly

    # Computing sideDir for the vertical line
    if b is None:  # G and B fall on a Vertical line, bx = gx
        ra = np.sign(rax - bx)  # for ra > 0
        rb = np.sign(rbx - bx)
        t1 = np.sign(a_tar[0] - bx)
        # t2 = np.sign(b_tar[0] - bx)
    # Computing sideDir for non vertical case
    else:  # Non-vertical line
        ya = m * rax + b  # vertical points' y that falls on th line just above or below robo A
        yb = m * rbx + b  # same for robo b  : these y's always have to be positive
        y_target1 = m * a_tar[0] + b  # same for
        y_target2 = m * b_tar[0] + b

        # we have positions w.r.t. line for all 4 points
        ra = np.sign(ya - ray)  # positive for
        rb = np.sign(yb - rby)
        t1 = np.sign(y_target1 - a_tar[1])
        t2 = np.sign(y_target2 - b_tar[1])


    # robots are on opposite sides
    if ra * rb < 0:  # robots are on opposite
        if ra * t1 > 0:  # rob a and targrt a same side:
            pass  # target for ra is a_tar, which is correct
        else:
            switch = True
            #a_tar, b_tar = b_tar, a_tar  # switch targets as they are currently assigned to bot on the opposite side

    # robot on the same side //
    elif ra * rb >= 0:  # both robots are on the same side, we need to measure dist from other target
        # and assign the other target to the robot closer to it

        if ra * t1 < 0 or rb * t1 < 0:  # robots are on opposite side as t1, then t1 is the farther target and calculate dist from it
            rob_a_dist_to_target = abs(rax - a_tar[0]) ** 2 + abs(ray - a_tar[1]) ** 2
            rob_b_dist_to_target = abs(rbx - a_tar[0]) ** 2 + abs(rby - a_tar[1]) ** 2

            if rob_b_dist_to_target < rob_a_dist_to_target:
                # rob b is closer to this target a, which needs reassignment
                switch = True
                #a_tar, b_tar = b_tar, a_tar


        else:  # robots are opposite of t2
            rob_a_dist_to_target = abs(rax - b_tar[0]) ** 2 + abs(ray - b_tar[1]) ** 2
            rob_b_dist_to_target = abs(rbx - b_tar[0]) ** 2 + abs(rby - b_tar[1]) ** 2

            if rob_b_dist_to_target > rob_a_dist_to_target:
                # rob a is closer to this target b, which is not currently assigned
                switch = True
                #a_tar, b_tar = b_tar, a_tar

    return switch

def calc_moveDir(ra, rb, a_tar, b_tar):
    # decide movement based on assigned targets
    rax, ray = ra[0], ra[1]
    rbx, rby = rb[0], rb[1]

    dax = int(np.sign(a_tar[0] - rax))
    day = int(np.sign(a_tar[1] - ray))

    dbx = int(np.sign(b_tar[0] - rbx))
    dby = int(np.sign(b_tar[1] - rby))
    return [dax, day], [dbx, dby]


# method to compute targets using goal nd bx posits, and predict move directions from robot and targets posits
def calc_targets_moveDirs(state):
    # rax, ray = robot_a
    # rbx, rby = robot_b

    rax, ray = state["robot_a"]
    rbx, rby = state["robot_b"]

    #bx, by = state["box"]
    #gx, gy = state["goal"]

    # SINGLE OR SEPARATE TARGET POINTS calculated FOR BOTH ROBOTS
    a_tar, b_tar = calc_targets(state["box"], state["goal"], isSingleF=False) # Targets CREATED

    #////////////////////////Assignment of targets to respective Robots
    #switch = False # So we can switch here instead of in the function
    switch = assign_targets(state, a_tar, b_tar) # Function tells us whether we need to reassign

    # switch if function says so
    if switch:
        a_tar, b_tar = b_tar, a_tar

    # decide movement based on assigned targets
    dax = int(np.sign(a_tar[0] - rax))
    day = int(np.sign(a_tar[1] - ray))

    dbx = int(np.sign(b_tar[0] - rbx))
    dby = int(np.sign(b_tar[1] - rby))

    # NEXT GET MOVE DIRS
    # distance from the targets to the robots. Take the sign as that's all that's needed to update position
    # dx = int(np.sign(rtx - rx))
    # dy = int(np.sign(rty - ry))

    return [dax, day], [dbx, dby], a_tar, b_tar, switch

def get_orientation(ta, tb, box):
    tax, tay = ta[0], ta[1]
    tbx, tby = tb[0], tb[1]
    bx = box[0]
    by = box[1]

    robot_a_orientation = 0
    robot_b_orientation = 0

    if tax - tbx == 0 or tay - tby == 0: # targets aligned on one axis
        if tax - tbx == 0:
            o = np.sign(bx - tax)
            if o < 0:
                robot_a_orientation = 180
                robot_b_orientation = 180
            else:
                robot_a_orientation = 0
                robot_b_orientation = 0
        else:
            o = np.sign(by - tay)
            # in a screen, to go up y is negative, the result of the above subtraction is opposite of a normal grid
            if o < 0:
                robot_a_orientation = 90
                robot_b_orientation = 90
            else:
                robot_a_orientation = 270
                robot_b_orientation = 270

    else: # targets are on different axis, so each one will be separate
        # robot a first
        if bx - tax > 0:
            robot_a_orientation = 0
        elif bx - tax < 0:
            robot_a_orientation = 180
        # in a screen, to go up y is negative, the result of the above subtraction is opposite of a normal grid
        elif by - tay > 0:
            robot_a_orientation = 270
        elif by - tay < 0:
            robot_a_orientation = 90

        # robot b
        if bx - tbx > 0:
            robot_b_orientation = 0
        elif bx - tbx < 0:
            robot_b_orientation = 180
        # in a screen, to go up y is negative, the result of the above subtraction is opposite of a normal grid
        elif by - tby > 0:
            robot_b_orientation = 270
        elif by - tby < 0:
            robot_b_orientation = 90

    robot_a_orientation, robot_b_orientation
    return robot_a_orientation, robot_b_orientation

def refine_orientation(robot_a_orientation, robot_b_orientation):
    dAngle = 36
    orient_pairs4_45Deg = [ (90, 0), (0, 270), (180, 90), (270, 180) ]
    orient_pairs4_45Deg_set = [frozenset(pair) for pair in orient_pairs4_45Deg]
    # Use current_pair for ordered input tuple
    current_pair = (robot_a_orientation, robot_b_orientation)
    current_pair_set = frozenset(current_pair)

    if current_pair_set in orient_pairs4_45Deg_set:
        idx = orient_pairs4_45Deg_set.index(current_pair_set)
        matched_pair = orient_pairs4_45Deg[idx]

        if current_pair == matched_pair:
            new_pair = [robot_a_orientation - dAngle, robot_b_orientation + dAngle]
        else:
            new_pair = [robot_a_orientation + dAngle, robot_b_orientation - dAngle]

        return new_pair

    return [robot_a_orientation, robot_b_orientation]


def obstacle_nav(robot_location, dir, target_location, obstacles):
    dir_x = dir[0]
    dir_y = dir[1]
    rx = robot_location[0]
    ry = robot_location[1]
    tx = target_location[0]
    ty = target_location[1]

    # check mistakes on the following
    # dir_x = -1
    # dir_y = -1
    # rx = 4
    # ry = 5
    # tx = 3
    # ty = 3

    ### Step 1: get which angle robot moving in

    # Compute the denominator r sqrt(x^2 + y^2)
    r = math.sqrt(dir_x ** 2 + dir_y ** 2)
    #Compute the angle in radians
    angle_radians = math.acos(dir_x / (r + 10e-12))
    # angle in degrees
    angle_dir = round( math.degrees(angle_radians) )
    # To get right angles in all quadrants
    if dir_y < 0:
        angle_dir = 360 - angle_dir


    ### Step 2: get if target is to left or right of robot by calculating angle to target in the same manner

    dx = tx - rx
    dy = ty - ry

    r_target = math.sqrt(dx ** 2 + dy ** 2)
    angle_radians = math.acos(dx / (r_target + 10e-12))
    angle_to_tar = round(math.degrees(angle_radians))
    if dy < 0:
        angle_to_tar = 360 - angle_to_tar

    # print(angle_to_tar)

    ### Step 3: explore options based on where target is

    rad = math.sqrt(2)  # radius defined here

    # target to the left of movement, add degrees to check; y flipped might make this be flipped too
    if angle_to_tar > angle_dir:
        angle_to_check = angle_dir + 45
        angle_to_check_rad = math.radians(angle_to_check) # inputs of trig functions should be in radians
        # get new directions by rcostheta rsintheta by rounding the output
        new_dir_x = round( rad * math.cos(angle_to_check_rad) )
        new_dir_y = round( rad * math.sin(angle_to_check_rad) )
        robot_next = (rx + new_dir_x, ry + new_dir_y)

        # print(robot_next)
        # print(obstacles)
        if robot_next not in obstacles:

            return (new_dir_x, new_dir_y)
        else:
            #print("Here", )
            angle_to_check = angle_dir + 90
            angle_to_check_rad = math.radians(angle_to_check)  # inputs of trig functions should be in radians
            # get new directisns by rcostheta r sintheta by rounding the output
            new_dir_x = round(rad * math.cos(angle_to_check_rad))
            new_dir_y = round(rad * math.sin(angle_to_check_rad))

            robot_next = (rx + new_dir_x, ry + new_dir_y)
            if robot_next not in obstacles:
                return (new_dir_x, new_dir_y)
            else:
                angle_to_check = angle_dir - 45
                angle_to_check_rad = math.radians(angle_to_check)  # inputs of trig functions should be in radians
                # get new directisns by rcostheta r sintheta by rounding the output
                new_dir_x = round(rad * math.cos(angle_to_check_rad))
                new_dir_y = round(rad * math.sin(angle_to_check_rad))

                robot_next = (rx + new_dir_x, ry + new_dir_y)
                if robot_next not in obstacles:
                    return (new_dir_x, new_dir_y)
                else:
                    angle_to_check = angle_dir - 90
                    angle_to_check_rad = math.radians(angle_to_check)  # inputs of trig functions should be in radians
                    # get new directisns by rcostheta r sintheta by rounding the output
                    new_dir_x = round(rad * math.cos(angle_to_check_rad))
                    new_dir_y = round(rad * math.sin(angle_to_check_rad))

                    #robot_next = (rx + new_dir_x, ry + new_dir_y)
                    return (new_dir_x, new_dir_y) # just return it


    # target is to the right of the original direction, subtract from original
    else:
        angle_to_check = angle_dir - 45
        #print(angle_to_check)
        angle_to_check_rad = math.radians(angle_to_check)  # inputs of trig functions should be in radians

        # get new directisns by rcostheta r sintheta by rounding the output
        # print(rad * math.cos(angle_to_check_rad))
        new_dir_x = round(rad * math.cos(angle_to_check_rad))
        new_dir_y = round(rad * math.sin(angle_to_check_rad))
        #print(new_dir_x, new_dir_y)
        robot_next = (rx + new_dir_x, ry + new_dir_y)

        if robot_next not in obstacles:
            return (new_dir_x, new_dir_y)
        else:
            angle_to_check = angle_dir - 90
            angle_to_check_rad = math.radians(angle_to_check)  # inputs of trig functions should be in radians
            # get new directisns by rcostheta r sintheta by rounding the output
            new_dir_x = round(rad * math.cos(angle_to_check_rad))
            new_dir_y = round(rad * math.sin(angle_to_check_rad))
            robot_next = (rx + new_dir_x, ry + new_dir_y)

            if robot_next not in obstacles:
                return (new_dir_x, new_dir_y)
            else:
                angle_to_check = angle_dir + 45
                angle_to_check_rad = math.radians(angle_to_check)  # inputs of trig functions should be in radians
                # get new directisns by rcostheta r sintheta by rounding the output
                new_dir_x = round(rad * math.cos(angle_to_check_rad))
                new_dir_y = round(rad * math.sin(angle_to_check_rad))
                robot_next = (rx + new_dir_x, ry + new_dir_y)

                if robot_next not in obstacles:
                    return (new_dir_x, new_dir_y)
                else:
                    angle_to_check = angle_dir + 90
                    angle_to_check_rad = math.radians(angle_to_check)  # inputs of trig functions should be in radians
                    # get new directisns by rcostheta r sintheta by rounding the output
                    new_dir_x = round(rad * math.cos(angle_to_check_rad))
                    new_dir_y = round(rad * math.sin(angle_to_check_rad))

                    robot_next = (rx + new_dir_x, ry + new_dir_y)
                    return (new_dir_x, new_dir_y) # just return for now



def query_simulator(state):
    # targets and actions calculated, along with if the targets were switched
    action_a, action_b, a_target, b_target, switch = calc_targets_moveDirs(state)

    # orientation?

    return { # returns a dict targets and move-dirs
        "robot_a": action_a,
        "robot_b": action_b,
        "target_a": [a_target[0], a_target[1]],
        "target_b": [b_target[0], b_target[1]],
        "switch": switch
        }










