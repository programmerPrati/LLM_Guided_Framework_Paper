from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


class SimplePolicy(nn.Module):
    def __init__(self, input_size, output_size=3): # output size needs to match number of outputs classifications
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            #nn.Softmax()
            #nn.Sigmoid(),  # Add sigmoid activation and BCE loss for classifying input to multiple classes
        )
        self.losses = []
        self.epoch_acc = []

    def forward(self, x):
        return self.fc(x)

    def train(self, policy, data_buffer, epochs=10):
        optimizer = optim.Adam(policy.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

        for _ in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            for state_tensor, action_idx in data_buffer:
                pred = policy(state_tensor)
                loss = criterion(pred.unsqueeze(0), torch.tensor(action_idx, dtype=torch.float).unsqueeze(0)) # dtype=torch.long
                epoch_losses.append(loss.item()) # append loss to graph

                predicted_class = torch.argmax(pred)
                true_class = torch.argmax(torch.tensor(action_idx))

                if predicted_class == true_class:
                    # This is a correct prediction
                    epoch_correct += 1
                epoch_total += 1

                # round_pred = torch.round(pred) # each value in tensor is rounded to 0 or 1
                # print("predict: ", round_pred)
                # print("label: ", torch.tensor(action_idx))
                # if torch.equal(round_pred, torch.tensor(action_idx).unsqueeze(0)): # label has dimension [1, 4], needs to be unsqueezed to match dim of pred [4]
                #     epoch_correct += 1
                # epoch_total += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # to plot accuracies over epoch
            single_epoch_acc = epoch_correct/epoch_total
            self.epoch_acc.append(single_epoch_acc)
            # to plot loss over epochs
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(avg_epoch_loss)  # Store average epoch loss


    def predict_action(self, policy, state_tensor):
        output = policy(state_tensor)  # Shape: [3] (e.g., tensor([0.5, -1.2, 0.8]))

        # Get the top value and its index (positions) from each output for x and y. highest output is prediction
        top_value, top_index = torch.topk(output, k=1)

        return int(top_index) # typecast to int as that's what we need, we are passing the index where the highest value occurred



    def plot_training_metrics(self):
        """Plot both training loss and accuracy curves. Call after training."""
        if not hasattr(self, 'losses') or not self.losses:
            raise ValueError("No losses recorded. Train the model first.")
        if not hasattr(self, 'epoch_acc') or not self.epoch_acc:
            raise ValueError("No accuracy recorded. Train the model first.")

        plt.figure(figsize=(20, 4))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss', color='blue', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_acc, label='Training Accuracy', color='green', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()








##////////////////////////////////////////////// Here Ends one NN ////////////////////////////////////////////////////////##
###########################################################################################################################


def custom_loss(logits, labels):
    # probs = torch.softmax(logits, dim=1)
    # true_dist = torch.zeros_like(probs).scatter(1, labels.unsqueeze(1), 1)

    probs = torch.sigmoid(logits)  # prediction, single value
    true_dist = labels.float() # convert label to float
    # Clamp probs to avoid numerical instability (still important)
    probs = probs.clamp(min=1e-7, max=1 - 1e-7)
    #print("label ", true_dist.shape, true_dist)
    #print("predictio: ", probs.shape, probs)
    #return (true_dist * (1 - probs)).sum(dim=1).mean
    ep = 1 # 0.1
    a = 1.2
    #return ep*((true_dist * (1 - probs**a)) + ( (1-true_dist) * (1 - (1-probs)**a) ))
    return true_dist * (1/(ep + probs**a) - 1/(1+ep)) + (1-true_dist) * (1/(ep + (1-probs)**a) - 1/(1+ep)) # .sum(dim=1).mean()


class assign_Targets_Policy(nn.Module):
    def __init__(self, input_size, output_size=1): # output size needs to match number of outputs classifications
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            #nn.ReLU(),
            nn.Linear(16, output_size),
            #nn.Sigmoid()
            # nn.Softmax()
        )
        self.optimizer = optim.SGD(self.parameters(), lr=0.001)
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        total = 60 + 40
        w0 = total / (2 * 60)  # for class 0
        w1 = total / (2 * 40)  # for class 1
        weights = torch.tensor([w0, w1], dtype=torch.float32)
        self.losses = [] # to plot losses
        self.epoch_acc = []
        #self.criterion = nn.CrossEntropyLoss() # weight = weights
        self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.MSELoss() # for linear regression
        #self.criterion = nn.BCELoss()  #  Binary Cross-Entropy with sigmoid

    def forward(self, x):
        # x = x/20
        return self.fc(x)


    def train(self, data, epochs=10):
        for epoch in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            for input, label in data:
                pred = self(input)  # shape: (1,) or scalar tensor
                label_tensor = torch.tensor([label], dtype=torch.float)  # shape: (1,), needed for single output
                loss = self.criterion(pred, label_tensor)
                # loss = custom_loss(pred, label_tensor)

                # to graph accuracy over epochs
                s_pred = np.where(pred >= 0, 1, -1) # if pred > 0  , 1, else -1
                if s_pred == label_tensor:
                    epoch_correct += 1
                epoch_total += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # to graph loss over epochs
                epoch_losses.append(loss.item())

            # to plot accuracies over epoch
            single_epoch_acc = epoch_correct/epoch_total
            self.epoch_acc.append(single_epoch_acc)
            # to plot loss over epochs
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(avg_epoch_loss)  # Store average epoch loss


    def predict(self, state_tensor):
        with torch.no_grad():  # No need to track gradients during prediction
            #self.eval()  # Set the network to evaluation mode
            pred = self(state_tensor)  # Shape: [output_size]
            #pred = torch.round(output)
            #print(rounded_output)
        return pred # typecast to int as that's what we need, we are passing the index where the highest value occurred


    def plot_training_metrics(self):
        """Plot both training loss and accuracy curves. Call after training."""
        if not hasattr(self, 'losses') or not self.losses:
            raise ValueError("No losses recorded. Train the model first.")
        if not hasattr(self, 'epoch_acc') or not self.epoch_acc:
            raise ValueError("No accuracy recorded. Train the model first.")

        plt.figure(figsize=(12, 4))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss', color='blue', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_acc, label='Training Accuracy', color='green', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

###################################################### Orientation NN starts now #####################

class orient_rob(nn.Module):
    def __init__(self, input_size, output_size=4): # output size needs to match number of outputs classifications
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, output_size),
            #nn.Sigmoid()
            nn.Softmax(dim=0)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=0.001)
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        total = 60 + 40
        w0 = total / (2 * 60)  # for class 0
        w1 = total / (2 * 40)  # for class 1
        weights = torch.tensor([w0, w1], dtype=torch.float32)
        self.losses = [] # to plot losses
        self.epoch_acc = []
        self.criterion = nn.CrossEntropyLoss() # weight = weights
        # self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.MSELoss() # for linear regression
        #self.criterion = nn.BCELoss()  #  Binary Cross-Entropy with sigmoid

    def forward(self, x):
        # x = x/20
        return self.fc(x)


    def train(self, data, epochs=10):
        for epoch in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            for input, label in data:
                pred = self(input)   # shape: [4]
                # label_tensor = torch.tensor([label], dtype=torch.float)
                # print(pred.shape)
                # print(label.shape)
                loss = self.criterion(pred, label.squeeze(0)) # label has dimension [1, 4], needs to be unsqueezed to match dim of pred [4]
                # loss = custom_loss(pred, label_tensor)

                # to graph accuracy over epochs
                round_pred = torch.round(pred) # each value in tensor is rounded to 0 or 1
                # print("predict: ", round_pred.shape)
                # print("label: ", label.shape)
                if torch.equal(round_pred, label.squeeze(0)): # label has dimension [1, 4], needs to be unsqueezed to match dim of pred [4]
                    epoch_correct += 1
                epoch_total += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # to graph loss over epochs
                epoch_losses.append(loss.item())

            # to plot accuracies over epoch
            single_epoch_acc = epoch_correct/epoch_total
            self.epoch_acc.append(single_epoch_acc)
            # to plot loss over epochs
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(avg_epoch_loss)  # Store average epoch loss


    def predict(self, state_tensor):
        with torch.no_grad():  # No need to track gradients during prediction
            #self.eval()  # Set the network to evaluation mode
            pred = self(state_tensor)  # Shape: [output_size]
            #pred = torch.round(output)
            #print(rounded_output)
        return pred # typecast to int as that's what we need, we are passing the index where the highest value occurred


    def plot_training_metrics(self):
        """Plot both training loss and accuracy curves. Call after training."""
        if not hasattr(self, 'losses') or not self.losses:
            raise ValueError("No losses recorded. Train the model first.")
        if not hasattr(self, 'epoch_acc') or not self.epoch_acc:
            raise ValueError("No accuracy recorded. Train the model first.")

        plt.figure(figsize=(12, 4))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss', color='blue', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_acc, label='Training Accuracy', color='green', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

class orient_rob(nn.Module):
    def __init__(self, input_size, output_size=4): # output size needs to match number of outputs classifications
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, output_size),
            #nn.Sigmoid()
            nn.Softmax(dim=0)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=0.001)
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        total = 60 + 40
        w0 = total / (2 * 60)  # for class 0
        w1 = total / (2 * 40)  # for class 1
        weights = torch.tensor([w0, w1], dtype=torch.float32)
        self.losses = [] # to plot losses
        self.epoch_acc = []
        self.criterion = nn.CrossEntropyLoss() # weight = weights
        # self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.MSELoss() # for linear regression
        #self.criterion = nn.BCELoss()  #  Binary Cross-Entropy with sigmoid

    def forward(self, x):
        # x = x/20
        return self.fc(x)


    def train(self, data, epochs=10):
        for epoch in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            for input, label in data:
                pred = self(input)   # shape: [4]
                # label_tensor = torch.tensor([label], dtype=torch.float)
                # print(pred.shape)
                # print(label.shape)
                loss = self.criterion(pred, label.squeeze(0)) # label has dimension [1, 4], needs to be unsqueezed to match dim of pred [4]
                # loss = custom_loss(pred, label_tensor)

                # to graph accuracy over epochs
                round_pred = torch.round(pred) # each value in tensor is rounded to 0 or 1
                # print("predict: ", round_pred.shape)
                # print("label: ", label.shape)
                if torch.equal(round_pred, label.squeeze(0)): # label has dimension [1, 4], needs to be unsqueezed to match dim of pred [4]
                    epoch_correct += 1
                epoch_total += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # to graph loss over epochs
                epoch_losses.append(loss.item())

            # to plot accuracies over epoch
            single_epoch_acc = epoch_correct/epoch_total
            self.epoch_acc.append(single_epoch_acc)
            # to plot loss over epochs
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(avg_epoch_loss)  # Store average epoch loss


    def predict(self, state_tensor):
        with torch.no_grad():  # No need to track gradients during prediction
            #self.eval()  # Set the network to evaluation mode
            pred = self(state_tensor)  # Shape: [output_size]
            #pred = torch.round(output)
            #print(rounded_output)
        return pred # typecast to int as that's what we need, we are passing the index where the highest value occurred


    def plot_training_metrics(self):
        """Plot both training loss and accuracy curves. Call after training."""
        if not hasattr(self, 'losses') or not self.losses:
            raise ValueError("No losses recorded. Train the model first.")
        if not hasattr(self, 'epoch_acc') or not self.epoch_acc:
            raise ValueError("No accuracy recorded. Train the model first.")

        plt.figure(figsize=(12, 4))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss', color='blue', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_acc, label='Training Accuracy', color='green', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

#////////////////////////////////////////////////////////////////////////////////////
class collision_det(nn.Module):
    def __init__(self, input_size, output_size=9): # output size needs to match number of outputs classifications
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 32),
            nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            #nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, output_size),
            nn.Sigmoid()
            # nn.Softmax(dim=0)
        )
        #self.optimizer = optim.SGD(self.parameters(), lr=0.001)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.losses = [] # to plot losses
        self.epoch_acc = []
        # self.criterion = nn.CrossEntropyLoss() # weight = weights
        # self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.MSELoss() # for linear regression
        self.criterion = nn.BCELoss()  #  Binary Cross-Entropy with sigmoid

    def forward(self, x):
        # x = x/20
        return self.fc(x)


    def train(self, data, epochs=10):
        for epoch in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            for input, label in data:
                pred = self(input)   # shape: [6]
                # label_tensor = torch.tensor([label], dtype=torch.float)

                loss = self.criterion(pred, label.squeeze(0)) # label has dimension [1, 6], needs to be unsqueezed to match dim of pred [4]
                # loss = custom_loss(pred, label_tensor)

                # to graph accuracy over epochs
                round_pred = torch.round(pred) # each value in tensor is rounded to 0 or 1
                # print("predict: ", round_pred.shape)
                # print("label: ", label.shape)
                if torch.equal(round_pred, label.squeeze(0)): # label has dimension [1, 6], needs to be unsqueezed to match dim of pred [6]
                    epoch_correct += 1
                epoch_total += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # to graph loss over epochs
                epoch_losses.append(loss.item())

            # to plot accuracies over epoch
            single_epoch_acc = epoch_correct/epoch_total
            self.epoch_acc.append(single_epoch_acc)
            # to plot loss over epochs
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(avg_epoch_loss)  # Store average epoch loss


    def predict(self, state_tensor):
        with torch.no_grad():  # No need to track gradients during prediction
            #self.eval()  # Set the network to evaluation mode
            pred = self(state_tensor)  # Shape: [output_size]
            #pred = torch.round(output)
            #print(rounded_output)
        return pred # typecast to int as that's what we need, we are passing the index where the highest value occurred


    def plot_training_metrics(self):
        """Plot both training loss and accuracy curves. Call after training."""
        if not hasattr(self, 'losses') or not self.losses:
            raise ValueError("No losses recorded. Train the model first.")
        if not hasattr(self, 'epoch_acc') or not self.epoch_acc:
            raise ValueError("No accuracy recorded. Train the model first.")

        plt.figure(figsize=(20, 4))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss', color='blue', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_acc, label='Training Accuracy', color='green', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def obstacle_nav_nn(self, robot_location, dir, target_location, obstacles):
        dir_x = dir[0]
        dir_y = dir[1]
        rx = robot_location[0]
        ry = robot_location[1]
        tx = target_location[0]
        ty = target_location[1]


        # Step 1: create the 9 directions to move around to explore (8 dirs + original dir
        # The output is an one-hot-vector, and each index represents the vector of the direction
        # 1-8, 45 deg each, start at north +45 and keep plusing 45
        # 9 is original dir
        # Step 2: Get next best position based on a set angle
        # collision_det()

        # Prepare input for NN
        input_tensor = torch.tensor([robot_location[0], robot_location[1], dir[0], dir[1], target_location[0], target_location[1]], dtype=torch.float32)
        nn_output = self.predict(input_tensor)

        # Divide the output
        isTargetPlusSide = round(nn_output[8].item()) #incresing angle side of the line of step direction
        other_outputs = nn_output[:8]

        # Extract angle_dir from the first 8 outputs
        predicted_class = torch.argmax(other_outputs).item()
        angle_dir = predicted_class * 45

        ### Step 3: explore options based on where target is
        rad = math.sqrt(2)  # radius defined here

        # target to the left of movement, add degrees to check; y flipped might make this be flipped too
        if isTargetPlusSide:
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
