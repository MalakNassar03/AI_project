from GeneticAlgo import assign_packages_with_genetic_algorithm
from simulated_annealing import simulated_annealing_main

print("************************ MENU ************************")
print("Please choose which algorithm to generate the solution:")
print("1-Simulated annealing")
print("2-Genetic algorithm")
print("3-exit")
choice = int(input("Enter choice:"))
while choice != 0:
    if choice == 1:
        print("************************ SIMULATED ANNEALING ************************")
        simulated_annealing_main()
    elif choice == 2:
        print("************************ GENETIC ALGORITHM ************************")
        assign_packages_with_genetic_algorithm()
    elif choice == 3:
        print("***")
        exit(0)
    else:
        print("Choice doesn't exist try again ")
    print("**********************************************************************")
    print("Please choose which algorithm to generate the solution:")
    print("1-Simulated annealing")
    print("2-Genetic algorithm")
    print("3-exit")
    choice = int(input("Enter choice:"))
