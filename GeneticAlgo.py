from collections import namedtuple
import random
import math
import matplotlib.pyplot as plt

vehicle_dict = {}
package_dict = {}
num_of_vehicles = 0
num_of_packages = 0
mutation_rate = 0.8
x_shop = 0
y_shop = 0
vehicles_capacity = 0
random.seed(42)  # Set a specific seed value (use any integer)

# Euclidean Distance = sqrt((X2-X1)^2 + (Y2-Y1)^2)
def euclidean_distance(X1, X2, Y1, Y2):
    return math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)

def fitness(solution):
    total_distance = 0  # initialize total distance
    for vehicle_id, package_list in solution.items():
        current_x = x_shop
        current_y = y_shop
        for p_id in package_list:
            p_data = package_dict[p_id]  # so we could use the named tuple
            total_distance += euclidean_distance(current_x, p_data.x, current_y, p_data.y)
            current_x = p_data.x
            current_y = p_data.y
        total_distance += euclidean_distance(x_shop, current_x, y_shop, current_y)
    return total_distance


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

def select_parents(population):
    # Implement a selection method that prioritizes solutions assigning all packages
    num_parents = min(len(population), 2 * (len(population) // 2))

    # Filter solutions that assign all packages correctly
    valid_solutions = [solution for solution in population if all(
        sum(package_dict[package].weight for package in assigned_packages) <= vehicles_capacity
        for assigned_packages in solution.values()
    )]

    if len(valid_solutions) < num_parents:
        # If there are not enough valid solutions, choose from the entire population
        parents = random.sample(population, num_parents)
    else:
        # Sort valid solutions by fitness and package count
        parents = sorted(valid_solutions, key=lambda x: (fitness(x), -sum(len(packages) for packages in x.values())))[:num_parents]

    return parents

def mutate(solution):
    unassigned_packages = list(package_dict.keys())

    # Inherit assignments from the parent
    child = {}
    for vehicle_id in solution.keys():
        child[vehicle_id] = solution[vehicle_id][:]

    # Keep track of assigned packages
    assigned_packages = set(package_id for packages in child.values() for package_id in packages)

    # Shuffle the unassigned packages to randomize the order of assignment
    random.shuffle(unassigned_packages)

    # Assign remaining packages to the child
    for package_id in unassigned_packages:
        package = package_dict[package_id]
        assigned = False

        for vehicle_id in child.keys():
            if package.weight <= vehicles_capacity - sum(package_dict[p].weight for p in child[vehicle_id]):
                child[vehicle_id].append(package_id)
                assigned = True
                assigned_packages.add(package_id)
                break

        if not assigned:
            break  # If no vehicle can accommodate the package, stop

    # Ensure that all packages are assigned correctly before returning
    if is_valid_solution(child):
        return child
    else:
        print("Mutation resulted in an invalid solution.")
        return solution


def crossover(parent1, parent2):
    child1 = {}
    child2 = {}

    # Inherit assignments from parents
    for vehicle_id in parent1.keys():
        child1[vehicle_id] = parent1[vehicle_id][:]
    for vehicle_id in parent2.keys():
        child2[vehicle_id] = parent2[vehicle_id][:]

    # Keep track of assigned packages
    assigned_packages1 = set(package_id for packages in child1.values() for package_id in packages)
    assigned_packages2 = set(package_id for packages in child2.values() for package_id in packages)

    # Assign remaining packages to children
    unassigned_packages1 = [package_id for package_id in list(package_dict.keys()) if package_id not in assigned_packages2]
    unassigned_packages2 = [package_id for package_id in list(package_dict.keys()) if package_id not in assigned_packages1]

    assign_remaining_packages(child1, unassigned_packages1, assigned_packages2)
    assign_remaining_packages(child2, unassigned_packages2, assigned_packages1)

    if is_valid_solution(child1) and is_valid_solution(child2):
        return child1, child2
    else:
        # If either of the crossover solutions is not valid, return the parents
        return parent1, parent2

def assign_remaining_packages(child, unassigned_packages, assigned_packages):
    for vehicle_id in child.keys():
        vehicle = vehicle_dict[vehicle_id]
        vehicle_packages = child[vehicle_id]
        current_capacity = sum(package_dict[p].weight for p in vehicle_packages)

        # Create a list to hold packages that couldn't be assigned due to capacity restrictions
        packages_not_assigned = []

        for package in unassigned_packages[:]:
            # Check if the package is not already assigned to any vehicle
            if package not in assigned_packages:
                package_data = package_dict[package]

                # Check if the package weight exceeds the vehicle's capacity
                if package_data.weight <= vehicles_capacity - current_capacity:
                    vehicle_packages.append(package)
                    vehicle_dict[vehicle_id].assigned_packages.append(package)
                    current_capacity += package_data.weight
                    assigned_packages.add(package)
                    unassigned_packages.remove(package)
                else:
                    packages_not_assigned.append(package)

        # Update the unassigned_packages array with the packages that couldn't be assigned due to capacity
        unassigned_packages.extend(packages_not_assigned)


def genetic_algorithm(population_size, num_generations, solution_limit):
    all_solutions = []  # Create a list to store all evaluated solutions

    for generation in range(num_generations):
        # Generate a new population for the current generation using random_solution_generator
        population = [random_solution_generator() for _ in range(population_size)]

        # Selection: Choose parents that assign all packages correctly
        parents = [solution for solution in population if is_valid_solution(solution)]

        # Check if there are enough parents for crossover
        if len(parents) < 2:
            continue

        # Crossover: Create children from parents that assign all packages correctly
        children = []
        for i in range(0, len(parents), 2):
            # Check if there are enough parents for the current crossover pair
            if i + 1 < len(parents):
                child1, child2 = crossover(parents[i], parents[i + 1])
                children.extend([child1, child2])

        # Mutation: Apply mutation to children
        for child in children:
            if random.random() < mutation_rate:
                mutated_child = mutate(child)
                if is_valid_solution(mutated_child):
                    children.append(mutated_child)

        # Ensure that children inherit the same vehicle assignments as their parents
        for child in children:
            for vehicle_id in parents[0]:
                if vehicle_id in parents[0]:
                    child[vehicle_id] = parents[0][vehicle_id]

        # Validate the solutions after mutation and crossover
        valid_children = [child for child in children if is_valid_solution(child)]

        # Store all evaluated solutions
        all_solutions.extend(valid_children)

        # Limit the number of solutions to the specified limit
        if len(all_solutions) > solution_limit:
            all_solutions = sorted(all_solutions, key=fitness)[:solution_limit]

        # Check if there are valid solutions in the current generation
        valid_solutions = [
            solution for solution in all_solutions if is_valid_solution(solution)
        ]

        # If valid solutions are found in the current generation, return them
        if valid_solutions:
            return valid_solutions

    # If no valid solutions are found after all generations, return an empty list
    return []


def is_valid_solution(solution):
    # Check if the solution violates any capacity constraints
    for assigned_packages in solution.values():
        if sum(package_dict[package].weight for package in assigned_packages) > vehicles_capacity:
            return False
    return True

def assign_packages_with_genetic_algorithm():
    global num_of_vehicles
    global num_of_packages
    global vehicles_capacity
    package_tuple = namedtuple("Package", ["weight", "x", "y"])
    vehicle_tuple = namedtuple("Vehicle", ["capacity", "assigned_packages"])
    num_of_vehicles = int(input("Enter the number of available vehicles: "))
    vehicles_capacity = int(input("Enter capacity limit for all vehicles: "))  # Update the global variable

    for i in range(num_of_vehicles):
        vehicle_dict[i + 1] = vehicle_tuple(vehicles_capacity, [])  # Include an empty list for assigned packages

    num_of_packages = int(input("Enter the number of packages ready to deliver: "))

    for i in range(num_of_packages):
        package_id = i + 1
        package_weight = int(input(f"Enter the weight of package {package_id}: "))
        x_coordinate = int(input(f"Enter x Coordinate of package {package_id}: "))
        y_coordinate = int(input(f"Enter y Coordinate of package {package_id}: "))
        package_dict[package_id] = package_tuple(package_weight, x_coordinate, y_coordinate)

    # Generate all the solutions
    all_solutions = genetic_algorithm(
        population_size=100, num_generations=100, solution_limit=30)

    if not all_solutions:
        print("No valid solutions found.")
        return

    # Initialize variables to track the best solution
    best_solution = None
    best_distance = float('inf')

    # Iterate through all generated solutions to find the best one
    for i, solution in enumerate(all_solutions, 1):
        total_distance = fitness(solution)
        if total_distance < best_distance:
            best_distance = total_distance
            best_solution = solution

    if not best_solution:
        print("No valid solutions found.")
        return

    print("***********************************************")
    print("Generated Solutions:")
    for i, solution in enumerate(all_solutions, 1):
        total_distance = fitness(solution)
        print("***********************************************")
        print(f"Solution {i}: Total Distance = {total_distance}")
        for vehicle_id, assigned_packages in solution.items():
            print(f"  Vehicle {vehicle_id}: Packages {assigned_packages}")

    # Print the top solution found
    print("\nTop Solution:")
    print(f"Total Distance = {best_distance}")
    for vehicle_id, assigned_packages in best_solution.items():
        print(f"Vehicle {vehicle_id}: Packages {assigned_packages}")

    # Create a list for packages that exceed the capacity limit
    unassigned_packages = []

    # Assign unassigned packages to the best solution
    for package_id in set(package_dict.keys()) - set(package_id for assigned_packages in best_solution.values()):
        package_weight = package_dict[package_id].weight
        assigned = False
        for vehicle_id, assigned_packages in best_solution.items():
            if sum(package_dict[p].weight for p in assigned_packages) + package_weight <= vehicles_capacity:
                assigned_packages.append(package_id)
                assigned = True
                break
        if not assigned:
            unassigned_packages.append(package_id)

    # Call the plot_solution function to visualize the best solution
    plot_solution(best_solution)  # Added this line to visualize the best solution

def plot_solution(solution):
    plt.figure(figsize=(8, 8))
    plt.title('Delivery Routes for Top Solution')

    # Plot shop location
    plt.scatter(x_shop, y_shop, color='red', label='Shop', s=100)

    # Plot package locations
    for package_id, package_data in package_dict.items():
        plt.scatter(package_data.x, package_data.y, label=f'Package {package_id}', s=50)

    # Create a consistent color mapping for vehicles
    vehicle_colors = {vehicle_id: random.choice(['b', 'g', 'c', 'm', 'y', 'k']) for vehicle_id in solution.keys()}

    for vehicle_id, assigned_packages in solution.items():
        x = [x_shop] + [package_dict[p].x for p in assigned_packages] + [x_shop]
        y = [y_shop] + [package_dict[p].y for p in assigned_packages] + [y_shop]
        color = vehicle_colors[vehicle_id]
        plt.plot(x, y, '->', label=f'Route for Vehicle {vehicle_id}', color=color)

    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.show()
