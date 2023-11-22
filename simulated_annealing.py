import math
import random
from collections import namedtuple

import matplotlib.pyplot as plt

vehicle_dict = {}  # package dictionary
package_dict = {}  # package dictionary
X_shop = 0  # X Coordinate of the delivery shop
Y_shop = 0  # Y Coordinate of the delivery shop
cooling_rate = 0.9
temperature = 1000
number_of_iterations = 100
assigned_packages = []  # assigned packages array


# ***********************************************************************************************
# VALUE ASSIGNMENT
def assign():
    global num_of_vehicles
    global num_of_packages
    global package_weight
    global vehicles_capacity

    # ***********************************************************************************************
    # Define a namedtuple to store vehicle data
    vehicle_tuple = namedtuple("Vehicle", ["capacity",
                                           "assigned_packages"])

    num_of_vehicles = int(input("Enter the number of available vehicles: \n"))  # enter number of available vehicles
    vehicles_capacity = int(
        input(f"Enter capacity limit for all vehicles: \n"))  # enter the capacity for all the vehicles
    # fill the vehicles dictionary
    for i in range(num_of_vehicles):
        vehicle = i + 1  # key of vehicles dictionary
        vehicle_data = vehicle_tuple(vehicles_capacity,
                                     assigned_packages)  # enter the namedtuple data to the vehicles dictionary
        vehicle_dict[vehicle] = vehicle_data

    # ***********************************************************************************************
    # Define a namedtuple to store package data
    package_tuple = namedtuple("Package", ["weight", "x",
                                           "y"])
    num_of_packages = int(input("Enter number of packages ready to deliver: \n"))  # enter number of available packages
    # fill the package's dictionary
    for i in range(num_of_packages):
        Package_ID = i + 1
        package_weight = int(input(f"Enter the weight of package {Package_ID}: \n"))  # enter the weight
        X_Coordinate = int(input(f"Enter x Coordinate of package {Package_ID}: \n"))  # enter X Coordinate
        Y_Coordinate = int(input(f"Enter y Coordinate of package {Package_ID}: \n"))  # enter Y Coordinate
        Package_Data = package_tuple(package_weight, X_Coordinate, Y_Coordinate)  # fill the namedtuble with the values
        package_dict[Package_ID] = Package_Data  # enter the namedtuple data to the package's dictionary
        data = package_dict[Package_ID]
        print("Weight:", data.weight)
        print("X-coordinate:", data.x)
        print("Y-coordinate:", data.y)


# ***********************************************************************************************
# Euclidean Distance= sqrt((X2-X1)^2 +(Y2-Y1)^2)
def euclidean_distance(X1, X2, Y1, Y2):
    return math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)


# ***********************************************************************************************
# evaluates the total cost of the route
def total_cost(solution):
    total_distance = 0  # initialize total distance

    # Iterate through each vehicle and its assigned package list in the 'solution' dictionary
    for vehicle, package_list in solution.items():
        current_x = X_shop  # Initialize the current x-coordinate to the shop's location
        current_y = Y_shop  # Initialize the current y-coordinate to the shop's location

        for p_id in package_list:
            p_data = package_dict[p_id]  # so we could use the namedtuple
            # Calculate the distance traveled from the current location to the package's location
            total_distance += euclidean_distance(current_x, p_data.x, current_y, p_data.y)
            # update the current location to the package's location
            current_x = p_data.x
            current_y = p_data.y
            # Calculate the distance traveled from the last package location back to the shop
        total_distance += euclidean_distance(X_shop, current_x, Y_shop, current_y)
    return total_distance


# ***********************************************************************************************
# RANDOM SOLUTION GENERATOR
def random_solution_generator():
    vehicle_keys_list = list(
        vehicle_dict.keys())  # create a list of the vehicles dictionary keys which is the vehicle number
    random.shuffle(vehicle_keys_list)  # shuffle the key list
    solution_dict = {vehicle_id: [] for vehicle_id in
                     vehicle_keys_list}  # Initialize a dictionary 'solution_dict' with vehicle IDs as keys and empty array for the assgined packages

    unassigned_packages_KeyList = list(
        package_dict.keys())  # create a list of the unassigned packages dictionary keys which is the packages ID
    random.shuffle(unassigned_packages_KeyList)  # shuffle the key list

    # Iterate through packages until all are assigned
    while unassigned_packages_KeyList:
        assignment_successful = False  # Flag to track if package was successfully assigned

        # Shuffle the vehicle list
        random.shuffle(vehicle_keys_list)

        for p_id in unassigned_packages_KeyList:  # iterate through unassigned_packages_KeyList
            p_data = package_dict[p_id]  # used to call the package's data

            for vehicle_id in vehicle_keys_list:  # iterate through vehicle_keys_list
                vehicle_capacity = vehicle_dict[
                    vehicle_id].capacity  # vehicle_capacity holds the capacity of the current vehicle
                total_weight_assigned = sum(package_dict[p].weight for p in solution_dict[
                    vehicle_id])  # calculates the total weight assigned to the current vehicle

                if vehicle_capacity >= total_weight_assigned + p_data.weight:  # checks if it is over the capacity
                    # if equal or under the capacity limit assign the package to the vehicle
                    solution_dict[vehicle_id].append(p_id)  # add to the solution dictionary
                    unassigned_packages_KeyList.remove(p_id)  # remove from available packages to assign
                    assignment_successful = True  # it was assigned successfully
                    break  # break out of the statement

            # so it doesn't enter an infinite loop and get stuck
            if assignment_successful:
                break

        if not assignment_successful:
            break

    return solution_dict  # return the solution dictionary


# ***********************************************************************************************
# SIMULATED ANNEALING ALGORITHM
def simulated_annealing():
    # initialize the initial solution
    current_solution = random_solution_generator()  # assign the solution to the randomly generated solution
    current_energy = total_cost(current_solution)  # the energy equals the total cost of the solution
    best_solution = current_solution  # assume that the current solution is the best solution
    best_energy = current_energy  # assume that the current energy is the best energy (cost)
    current_temperature = temperature  # the current temp=1000
    all_solutions = []  # Store all solutions in an array
    # iterate 100 times
    for _ in range(number_of_iterations):
        neighbor_solution = random_solution_generator()  # Generate a neighboring solution using the random solution generator
        neighbor_energy = total_cost(neighbor_solution)  # calculate the energy(cost) of the neighbor_solution

        delta_energy = neighbor_energy - current_energy  # calculate the delta energy

        # If delta_energy is negative (a lower energy state is reached) or if a random probability is less
        # than the acceptance probability based on temperature and energy difference,
        # accept the neighbor solution
        # expansion
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / current_temperature):
            # update the current_solution and current_energy to neighbor's value
            current_solution = neighbor_solution
            current_energy = neighbor_energy
        # Update the best solution if the current solution has lower energy then best energy that was initialized too
        if current_energy < best_energy:
            best_solution = current_solution
            best_energy = current_energy

        current_temperature *= cooling_rate  # reduce the temperature
        all_solutions.append(current_solution.copy())  # Store the current solution

    return best_solution, best_energy, all_solutions


# ***********************************************************************************************

def plot_solution(best_solution):
    plt.figure(figsize=(8, 8))
    # Mark the shop location as a red square marker
    plt.scatter(X_shop, Y_shop, c='r', marker='s', label='Shop')

    # Iterate through vehicles and their assigned packages
    for vehicle_id, package_list in best_solution.items():
        x_values = [X_shop]  # Initialize x-values with the shop's location
        y_values = [Y_shop]  # Initialize y-values with the shop's location

        for p_id in package_list:
            p_data = package_dict[p_id]
            x_values.append(p_data.x)  # Append the package's x-coordinate
            y_values.append(p_data.y)  # Append the package's y-coordinate

        # Return to the shop after delivering all packages
        x_values.append(X_shop)
        y_values.append(Y_shop)
        # Plot the vehicle's route
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=f'Vehicle {vehicle_id}')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    # Set the plot's title
    plt.title('Best Solution - Vehicle Routes')
    plt.grid(True)
    # Show the plot
    plt.show()


# ***********************************************************************************************


def simulated_annealing_main():
    assign()
    best_solution, best_distance, all_solutions = simulated_annealing()

    # Print all possible solutions
    for i, solution in enumerate(all_solutions):
        print(f"Solution {i + 1} (Vehicle assignments):", solution)
        print("Total distance:", total_cost(solution))
    print("Best solution (Vehicle assignments):", best_solution)
    print("Total distance of best solution:", best_distance)
    # plot_solution(best_solution)
