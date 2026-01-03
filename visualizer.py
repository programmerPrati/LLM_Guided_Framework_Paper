import pygame
import math

def draw_grid(state, grid_size, screen_size=700):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("GridWorld")

    # Calculate cell size and margin to perfectly fit the grid
    # Total margin space is (grid_size + 1) * margin (margin around each cell and around the grid)
    # Let's make margin half of cell_size for good proportions
    # screen_size = grid_size * cell_size + (grid_size + 1) * margin
    # Let margin be 0
    # Then: screen_size = grid_size * cell_size + (grid_size + 1) * cell_size/4
    # screen_size = cell_size * (grid_size)
    # cell_size = screen_size / (grid_size)
    cell_size = screen_size / grid_size

    margin = 0

    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)

    GOAL_COLOR = (255, 135, 110) # red sahde fr goal

    ROBOT_A_COLOR = (90, 200, 220) # green for robo A
    ROBOT_B_COLOR = (90, 150, 255) # Blue for B
    TARGET_A_COLOR = (235, 255, 225)  # Light green
    TARGET_B_COLOR = (220, 235, 255)  # Light blue

    BOX_COLOR =  (225, 188, 128) #(240, 160, 0) box is goldenish brown

    OBSTACLE_COLOR = (64, 64, 64)



    robot_a = state["robot_a"]
    robot_b = state["robot_b"]
    box = tuple(map(int, state["box"]))
    goal = tuple(map(int, state["goal"]))
    target_a = state["target_a"]  # Get target_a
    target_b = state["target_b"]  # Get target_b

    robot_a_direction = state["robot_a_orient"]
    robot_b_direction = state["robot_b_orient"]

    obstacles = state["obstacles"]

    # print("robot_a_direction", robot_a_direction)
    # print("robot_b_direction", robot_b_direction)

    screen.fill(WHITE)

    # # Draw grid
    # for y in range(grid_size):
    #     for x in range(grid_size):
    #         rect = pygame.Rect(
    #             x * (cell_size + margin) + margin,
    #             y * (cell_size + margin) + margin,
    #             cell_size,
    #             cell_size
    #         )
    #         pygame.draw.rect(screen, GRAY, rect, 1)

    # Draw goal
    ovr = 6 # take even numbers here
    pygame.draw.rect(screen, GOAL_COLOR, pygame.Rect(
        goal[0] * (cell_size + margin) + margin - ovr / 2,
        goal[1] * (cell_size + margin) + margin - ovr / 2,
        cell_size+ovr, cell_size+ovr
    ))

    # Draw box
    pygame.draw.rect(screen, BOX_COLOR, pygame.Rect(
        box[0] * (cell_size + margin) + margin,
        box[1] * (cell_size + margin) + margin,
        cell_size, cell_size
    ))

    # Draw target highlights if they exist
    # if target_a is not None:
    #     pygame.draw.rect(screen, TARGET_A_COLOR, pygame.Rect(
    #         target_a[0] * (cell_size + margin) + margin,
    #         target_a[1] * (cell_size + margin) + margin,
    #         cell_size, cell_size
    #     ))

    # if target_b is not None:
    #     pygame.draw.rect(screen, TARGET_B_COLOR, pygame.Rect(
    #         target_b[0] * (cell_size + margin) + margin,
    #         target_b[1] * (cell_size + margin) + margin,
    #         cell_size, cell_size
    #     ))


    # obstacles
    for o in obstacles:
    #if obs1 is not None:
        pygame.draw.rect(screen, OBSTACLE_COLOR, pygame.Rect(
            o[0] * (cell_size + margin) + margin,
            o[1] * (cell_size + margin) + margin,
            cell_size, cell_size
        ))





    # draw robots with the stick

    # robot a #####################
    rbSizeFact = 2.75
    notch_thickness = 5
    radius = cell_size // rbSizeFact
    notch_length = radius * 4  # Slightly longer than the radius

    if robot_a_direction is not None:
        center_ax = robot_a[0] * (cell_size + margin) + margin + cell_size // 2
        center_ay = robot_a[1] * (cell_size + margin) + margin + cell_size // 2

        # Draw the main circle
        pygame.draw.circle(screen, ROBOT_A_COLOR, (center_ax, center_ay), radius)
        # Calculate notch position (a line extending from the circle)
        angle_rad = math.radians(robot_a_direction)
        # Calculate end point of the notch
        end_x = center_ax + notch_length * math.cos(angle_rad)
        end_y = center_ay - notch_length * math.sin(angle_rad)  # Negative because Pygame y increases downward
        # Draw the notch line
        pygame.draw.line(screen, ROBOT_A_COLOR, (center_ax, center_ay), (end_x, end_y), notch_thickness)
    else:
        pygame.draw.circle(screen, ROBOT_A_COLOR, (
            robot_a[0] * (cell_size + margin) + margin + cell_size // 2,
            robot_a[1] * (cell_size + margin) + margin + cell_size // 2
        ), radius)




    # robot b: ####################

    if robot_b_direction is not None:
        # Calculate Robot B circle center
        center_bx = robot_b[0] * (cell_size + margin) + margin + cell_size // 2
        center_by = robot_b[1] * (cell_size + margin) + margin + cell_size // 2

        # Draw Robot B's main circle
        pygame.draw.circle(screen, ROBOT_B_COLOR, (center_bx, center_by), radius)
        # Calculate notch position for Robot B
        angle_rad_b = math.radians(robot_b_direction)
        # Calculate end point of Robot B's notch
        end_bx = center_bx + notch_length * math.cos(angle_rad_b)
        end_by = center_by - notch_length * math.sin(angle_rad_b)
        # Draw Robot B's notch line
        pygame.draw.line(screen, ROBOT_B_COLOR, (center_bx, center_by), (end_bx, end_by), notch_thickness)
    else:
        pygame.draw.circle(screen, ROBOT_B_COLOR, (
            robot_b[0] * (cell_size + margin) + margin + cell_size // 2,
            robot_b[1] * (cell_size + margin) + margin + cell_size // 2
        ), radius)



    # pygame.draw.circle(screen, ROBOT_A_COLOR, (
    #     robot_a[0] * (cell_size + margin) + margin + cell_size // 2,
    #     robot_a[1] * (cell_size + margin) + margin + cell_size // 2
    # ), cell_size // 3)

    # pygame.draw.circle(screen, ROBOT_B_COLOR, (
    #     robot_b[0] * (cell_size + margin) + margin + cell_size // 2,
    #     robot_b[1] * (cell_size + margin) + margin + cell_size // 2
    # ), cell_size // 2.85)

    SKIP_BUTTON_COLOR = (100, 100, 100)  # Dark gray
    TEXT_COLOR = (255, 255, 255)  # White
    font_size = 30
    font = pygame.font.Font(None, font_size)
    skip_button_rect = pygame.Rect(10, 10, 100, 40)  # x, y, width, height
    pygame.draw.rect(screen, SKIP_BUTTON_COLOR, skip_button_rect)
    skip_text = font.render("Skip", True, TEXT_COLOR)
    text_rect = skip_text.get_rect(center=skip_button_rect.center)
    screen.blit(skip_text, text_rect)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if skip_button_rect.collidepoint(event.pos):
                return True
    return False