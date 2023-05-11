import math
import numpy as np
from math import gcd


################### Level 1 ###################
def solution(string):
    # Define the list to be used for the replacement
    list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    # Loop through the string and replace the characters if they are in the list
    for k in range(len(string)):
        character = string[k]
        if character in list:
            # Define the character to be replaced and splice the string
            replace = list[-list.index(character)-1]
            string = string[:k] + replace + string[k+1:]
    # Return the string
    return string


################### Level 2, Problem 1 ###################
def elevator(l):
    # Import numpy and initialize the sorted array
    import numpy as np
    sorted = np.zeros((0,3), dtype = int)

    # Loop through the list of beta versions and split them into a list of integers (vector)
    for k in range(len(l)):
        vector = np.array(l[k].split("."), dtype = int)

        # If the vector is not 3 elements long, append -1's to the end of the vector
        if len(vector) == 1:
            vector = np.append(vector, [-1,-1])
        if len(vector) == 2:
            vector = np.append(vector, -1)
        
        # Initialize a position counter and stop value. Loop through all three positions of the vector
        k = 0
        stop = len(sorted)
        for j in range(3):
            
            # Assign the start value and loop through the segmented sorted array
            start = k
            for i in range(len(sorted[start:stop])):

                # If vector entry is greater than the sorted array entry, increment the counter
                if vector[j] > sorted[start:stop,j][i]:
                    k += 1

                # If less than, break the loop and update the stop value
                else:
                    stop = k
                    # update the stop value with a while loop
                    while vector[j] == sorted[:,j][stop]:
                        stop += 1
                        # break the loop if the end of the array is reached
                        if len(sorted[:,0]) == stop:
                            break
                    break

        # Insert the vector into the sorted array at position k
        sorted = np.insert(sorted, k, vector, axis = 0)
    
    # Initialize the return list and loop through the sorted array
    returnlist = []
    for k in range(len(sorted)):

        # Discard all -1's by looping through the vector and appending the non -1 values to a list
        betaNum = []
        for j in range(3):

            # discard all -1's and break the loop
            if sorted[k][j] == -1:
                break

            # otherwise append the value to betaNum
            else:
                betaNum.append(sorted[k][j])
        
        # Join the list of integers into a string and append it to the return list
        returnlist.append(".".join(str(x) for x in betaNum))

    # Return the return list
    return returnlist


################### Level 2, Problem 2 ###################

def gears(pegs):
    # Import numpy and initialize the difference array
    import numpy as np
    difference = np.zeros(len(pegs), dtype = float)

    # Loop through the pegs and calculate the difference between each element for back substitution
    for i in range(1, len(pegs)):
        difference[i] = pegs[i] - pegs[i-1] - difference[i - 1]
    
    # If the length of the pegs is even, divide the last element by 3 and set the addVar to the last element
    if len(pegs) % 2 == 0:
        difference[-1] = difference[-1] / 3.0
        addVar = difference[-1]

    # If the length of the pegs is odd, multiply the last element by -1 and set the addVar to the last element
    else:
        difference[-1] =  - difference[-1]
        addVar = difference[-1]

    # if the addVar is not a positive integer greater than 1, return [-1,-1]
    if addVar < 1:
        return [-1,-1]

    # Loop through the difference array and back substitute the values
    for i in range(len(pegs)-2, -1, -1):
        difference[i] = difference[i] + (-1)**(i) * 2 * addVar

        # If the difference array contains any value less than 1, return [-1,-1]
        if difference[i] < 1:
            return [-1,-1]

    # Check if the first gear is a whole number and set numerator and denominator accordingly
    if difference[0] == round(difference[0]):
        num = int(difference[0])
        den = 1

    # If not, multiply the first gear by 3 and set the numerator and denominator accordingly
    else:
        num = int(difference[0] * 3)
        den = 3

    # Return the numerator and denominator
    return [num,den]

################### Level 3, Problem 1 ###################

def pellets(n):
    """This function wants to find the most efficient way to get to 1 from a given number n.
    We must utilize Bellman's optimization principle to find the most efficient way to get to 1.
    We do this by using a top-down approach and memoization to store the values of the subproblems,
    in order to avoid repeated calculations and improve the efficiency of the algorithm."""
    # Initialize the dictionary and convert n to an integer
    dict = {1:0}
    n = int(n)

    # Define our recursion function to help with memoization
    def recursion(n):
        # If the number is in the dictionary, return the value
        if n in dict.keys():
            return dict[n]

        # If the number is not in the dictionary, initialize k and a test counter
        else:
            testCounter = 0
            k = n

            # Loop through the number until it is even and increment the test counter
            while k % 2 == 0:
                k = k / 2
                testCounter += 1

            # Base case: if k is 1, count is the test counter
            if k == 1:
                count = testCounter

            # Otherwise, count is the minimum of the recursion of k-1 and k+1, plus the test counter and 1
            else:
                count = min(recursion(k-1), recursion(k+1)) + testCounter + 1
            
            # Add the key and value to the dictionary and return the count
            dict[n] = count
            return count
    
    # Return the recursion function on n
    return recursion(n)


################### Level 3, Problem 2 ###################

def escape(map):
    """make a breadth first search from position 0,0 and return the number of steps"""
    # get the number of rows and columns and import numpy
    import numpy as np
    colNum = len(map[0])
    rowNum = len(map)

    # Define a function that steps through the map and updates the distances and boundary arrays
    def stepCount(queue, distances, boundary, rowPoint, colPoint, rowDir, colDir):
        """This function takes in our distances array, whih is a matrix marking which cells have been visited,
        as well as their distance, and the boundary array, which is a matrix marking which cells are on the boundary.
        We then input a current point, and a direction for it to step through. It updates the distances and boundary
        accordingly"""
        # Mark the adjacent cell as a boundary and check if it a wall or a cell that is traversable
        boundary[rowPoint + rowDir,colPoint + colDir] = True
        if (map[rowPoint + rowDir][colPoint + colDir] == 0):

            # If the cell is traversable, check if it has been visited, and then it to the queue and update the distances
            if distances[rowPoint + rowDir,colPoint + colDir] == 0:
                queue.append([rowPoint + rowDir,colPoint + colDir])
                distances[rowPoint + rowDir,colPoint + colDir] = distances[rowPoint,colPoint] + 1

    # Make a function that does the breadth first search on the map
    def BFS(queue, map, distances, boundary):
        # Loop through the queue until it is empty
        while len(queue) != 0:
            # Pop the first element from the queue, and add it to the marked list
            node = queue.pop(0)
            rowPoint, colPoint = node[0], node[1]

            # Add below point to the queue if it is not marked and not the end
            if (rowPoint < len(map) - 1):
                stepCount(queue, distances, boundary, rowPoint, colPoint, 1, 0)
            
            # Add above point to the queue if it is not marked and not the end
            if (rowPoint > 0):
                stepCount(queue, distances, boundary, rowPoint, colPoint, -1, 0)

            # Add right point to the queue if it is not marked and not the end
            if (colPoint < len(map[0]) - 1):
                stepCount(queue, distances, boundary, rowPoint, colPoint, 0, 1)
            
            # Add left point to the queue if it is not marked and not the end
            if (colPoint > 0):
                stepCount(queue, distances, boundary, rowPoint, colPoint, 0, -1)

    # Initialize our queue and the distances array and set the start point to 1
    queue = [[0,0]]
    distances = np.zeros((rowNum,colNum), dtype = int)
    distances[0,0] = 1
    
    # Initialize the boundary array and set the start point to True
    boundary = np.full((rowNum,colNum), False, dtype = bool)
    boundary[0,0] = True

    # Run the BFS function on the map and copy the distances and boundary arrays
    BFS(queue, map, distances, boundary)
    topDistances = distances.copy()
    topBoundary = boundary.copy()

    # Initialize a new queue and distances array and set the end point to 1
    queue = [[rowNum - 1,colNum - 1]]
    distances = np.zeros((rowNum,colNum), dtype = int)
    distances[rowNum - 1,colNum - 1] = 1

    # Initialize a new boundary array and set the end point to True
    boundary = np.full((rowNum,colNum), False, dtype = bool)
    boundary[rowNum - 1,colNum - 1] = True

    # Run the BFS function on the map and copy the distances and boundary arrays
    BFS(queue, map, distances, boundary)
    bottomDistances = distances.copy()
    bottomBoundary = boundary.copy()

    # Make a test matrix that is the product of the top and bottom boundary arrays, initialize a candidates list
    testMatrix = topBoundary * bottomBoundary
    candidates = []

    # Loop through the test matrix and add the points to the candidates list
    for i in range(rowNum):
        for j in range(colNum):

            # If the point is in the test matrix, initialize the top and bottom test arrays for comparison
            if testMatrix[i,j] == True:
                topTestArray = []
                bottomTestArray = []

                """Run through the top array and add the values to the top test array 
                if adjacent cells are walls and are not outsie of the map. Taking the minimum 
                of the top test array shows the minimum number of steps to get to the point"""
                # Check if it is not the top or bottom row and append the values to top array if they are walls
                if i > 0 and topDistances[i - 1,j] != 0:
                    topTestArray.append(topDistances[i - 1,j])
                if i < rowNum - 1 and topDistances[i + 1,j] != 0:
                    topTestArray.append(topDistances[i + 1,j])

                # Check if it is not the left or right column and append the values to top array if they are walls
                if j > 0 and topDistances[i,j - 1] != 0:
                    topTestArray.append(topDistances[i,j - 1])
                if j < colNum - 1 and topDistances[i,j + 1] != 0:
                    topTestArray.append(topDistances[i,j + 1])

                """Run through the bottom array and add the values to the bottom test array 
                if adjacent cells are walls and are not outsie of the map. We take the minimum
                again to find the shortest path between the point and the end""" 
                # Check if it is not the top or bottom row and append the values to bottom array if they are walls 
                if i > 0 and bottomDistances[i - 1,j] != 0:
                    bottomTestArray.append(bottomDistances[i - 1,j])
                if i < rowNum - 1 and bottomDistances[i + 1,j] != 0:
                    bottomTestArray.append(bottomDistances[i + 1,j])

                # Check if it is not the left or right column and append the values to bottom array if they are walls
                if j > 0 and bottomDistances[i,j - 1] != 0:
                    bottomTestArray.append(bottomDistances[i,j - 1])
                if j < colNum - 1 and bottomDistances[i,j + 1] != 0:
                    bottomTestArray.append(bottomDistances[i,j + 1])
                
                # If both arrays are not empty, add the minimum values of top and bottom, and append to candidates list
                if len(topTestArray) != 0 and len(bottomTestArray) != 0:
                    candidates.append(min(topTestArray) + min(bottomTestArray) + 1)
    
    # return the minimum value of the candidates list, this is the shortest path with removing a wall
    return min(candidates)

#################### Level 3, Problem 3 ####################

def gcdMultiple(list):
        """This function takes a list of numbers and returns the gcd of the list"""
        # Initialize the previous value to the first value in the list
        prev = int(list[0])
        for i in range(1,len(list)):
            # Get the gcd of the previous value and the next value in the list, and return the gcd
            prev = math.gcd(prev, int(list[i]))
        return prev

def solution(m):
    """We are using an absorbing Markov chain to find the probability of being in each terminal state
    in order to solve this problem. By calculating the fundamental matrix, we can find the probability of
    starting in the initial state by taking the first row of the fundamental matrix. We calculate this matrix
    by taking the identity matrix, subtracting the transition matrix, and then taking the inverse of the
    resulting matrix. We must take care to work using rational numbers."""
    # initialize the denominator array and the zero rows counter, and get size
    denomArray = np.array([],dtype=int)
    zeroRows = 0
    height = len(m)
    endState = [not any(row) for row in m]

    # Loop through the matrix and get the denominators of each row and append to the denominator array
    for i in range(height):
        denom = int(round(sum(m[i])))

        # If the denominator is zero, increment the zero rows counter
        if denom == 0:
            zeroRows += 1

            # Set the denominator to 1 and append to the denominator array
            denom = 1
        denomArray = np.append(denomArray, denom)

    # Make the resulting matrix which we will take the inverse of
    m = np.diag(denomArray) - m

    """Helper Functions"""
    # Define a function that simplifies a row
    def simplifyRow(Q,denomArray, row):
        """This function takes a matrix, a denominator array, and a row number, and simplifies the row"""
        # Get the gcd of the row and the denominator, and simplify both
        factor = gcdMultiple(np.append(Q[row],denomArray[row]))
        Q[row] = Q[row] / factor
        denomArray[row] = denomArray[row] / factor

    # Define a function to get the rational inverse of a matrix
    def rationalInverse(Q):
        """This function takes a matrix of integers, turns it into a markov chain,
        and returns the numerators of the inverse of the matrix, and its denominator"""
        # Make Q a numpy array, make a diagonal matrix with the denominators, and concatenate the two. Get the size
        Q = np.array(Q, dtype=int)
        diag = np.diag(denomArray)
        Q = np.concatenate((Q, diag), axis=1)
        height = len(Q)
    
        # Loop through the matrix, doing gaussian elimination. Get the pivot value.
        for i in range(height):
            pivot = Q[i][i]
            for j in range(i+1,height):
                simplifyRow(Q,denomArray,j)
                # If the value is zero, continue
                if Q[j][i] == 0:
                    continue
                
                # Adjust the denominator values by the pivot value, and row reduce the matrix
                denomArray[j] = denomArray[j] * pivot
                simplifyRow(Q,denomArray,j)
                Q[j] = (pivot * Q[j] - Q[j][i] * Q[i])

            # Multiply the row by the denominator value and the denominator by the pivot value to get the pivot to 1
            Q[i] = Q[i] * denomArray[i]
            denomArray[i] = denomArray[i] * pivot

        # Repeat this process, but backwards, to get the inverse of the matrix
        for i in range(height-1,-1,-1):
            pivot = Q[i][i]
            for j in range(i-1,-1,-1):
            
                # If the value is zero, continue
                if Q[j][i] == 0:
                    continue

                # Adjust the denominator values by the pivot value, and row reduce the matrix
                denomArray[j] = denomArray[j] * pivot
                simplifyRow(Q,denomArray,j)
                Q[j] = (pivot * Q[j] - Q[j][i] * Q[i])
        # Return the first row of the inverse of the matrix and the first denominator entry
        return Q[0,height:], denomArray[0]

    # Get the inverse of the matrix and add it to the answer array
    array,denominator = rationalInverse(m)
    answer = np.array([],dtype=int)
    
    # Loop through the array and append the numerator to the answer array
    for i in range(len(array)):
        if endState[i]:
            answer = np.append(answer, array[i])

    # Append the denominator to the array and return the array
    answer = np.append(answer, denominator)
    factor = gcdMultiple(answer)
    return answer / factor


######################## Level 4, problem 1 ########################

def extend(entrances, exits, path):
    """Extend the path matrix by including super source and super sink nodes. 
    This will handle the case where there are multiple entrances and exits.
    inputs:
        entrances: list of entrance nodes - ndarray of shape (n,)
        exits: list of exit nodes - ndarray of shape (k,)
        path: path matrix - ndarray of shape (l, l)
        
    outputs:
        newPath: extended path matrix - ndarray of shape (l+2, l+2)"""
    # Get the number of nodes, initialize the new path matrix and augment the entrances and exits
    n = len(path)
    newPath = np.zeros((n+2, n+2))
    entrances = np.array(entrances) + 1
    exits = np.array(exits) + 1

    # Fill in the new path matrix with the old path matrix and then add the super source and super sink nodes
    newPath[1:-1, 1:-1] = path
    newPath[0, entrances] = np.inf
    newPath[exits, -1] = np.inf
    
    # return the extended path matrix as a list of lists
    return newPath


def bfs(capacities, currentFlow):
    """Run a breadth first search to return the layer array.
    inputs:
        capacities: capacity matrix - ndarray of shape (n, n)
        currentFlow: current flow matrix - ndarray of shape (n, n)
    
    outputs:
        layers: layer array - ndarray of shape (n,)"""
    # Get number of nodes, initialize layers array, initialize the queue, and initialize the considered array
    n = len(capacities)
    layers = np.zeros(n, dtype=int)
    queue = [(-1,0)]
    considered = np.zeros(n)

    # While the queue is not empty, run the breadth first search. Pop the node, update layers, and find next steps
    while queue:
        prevNode, node = queue.pop()
        layers[node] = layers[prevNode] + 1
        nextStep = np.where((capacities[node] > 0) & (currentFlow[node] < capacities[node]))[0]
        
        # For each next step, if the node has not been considered, insert it into the queue
        for adjacent in nextStep:
            if considered[adjacent] == 0:
                queue.insert(0,(node,adjacent))
                considered[adjacent] = 1

    # Return the layer nodes
    return layers


#Start from the back and use the level graph to get the path to the super source. Then find the values where the left is 1 less than the right. Take those values in a list. Index the queue using those values 
def dfs(capacities, currentFlow, layers):
    """Run a depth first search to return the updated current flow matrix.
    inputs:
        capacities: capacity matrix - ndarray of shape (n, n)
        currentFlow: current flow matrix - ndarray of shape (n, n)
        layers: layer array - ndarray of shape (n,)
        
    outputs:
        currentFlow: updated current flow matrix - ndarray of shape (n, n)"""
    # Get nodes number, initialize traverse array, queue, path, and considered array
    n = len(capacities)
    traverse = capacities > 0
    queue = [0]
    path = []
    considered = np.zeros(n)
    
    # While the queue is not empty, run the depth first search and Pop each node
    while queue:
        node = queue.pop()

        # If the node backtracks, remove the backtrack from the path and append the node
        if layers[node] - 1 != len(path):
            j = path[layers[node] - 2]
            k = path[layers[node] - 1]
            traverse[j, k] = False
            path = path[:layers[node] - 1]
        path.append(node)
        
        # If the node is the super sink, update the current flow matrix, reset the queue, path, and considered array
        if node == n-1:
            currentFlow += newflow(capacities, currentFlow, path)
            queue = [0]
            path = []
            considered = np.zeros(n)
            continue

        # If the node is not the super sink, find the next steps and append them to the queue if they have not been considered
        nextStep = np.where((traverse[node] == True) & (currentFlow[node] < capacities[node]))[0]
        for adjacent in nextStep:
            if (layers[adjacent] == layers[node] + 1) and (considered[adjacent] == 0):
                queue.append(adjacent)
                considered[adjacent] = 1

    # Return the updated current flow matrix
    return currentFlow

def newflow(capacaties, currentFlows, path):
    """Update the current flow matrix given the path.
    inputs:
        capacities: capacity matrix - ndarray of shape (n, n)
        currentFlow: current flow matrix - ndarray of shape (n, n)
        path: path matrix - ndarray of shape (l, l)
    
    outputs:
        currentFlow: updated current flow matrix - ndarray of shape (n, n)
    """
    # Get the length of path, initialize flow matrix, and initialize flow list
    k = len(path)
    flows = np.zeros_like(capacaties)
    flowlist = []

    # Loop through each level, and indicate forwards or backwards flow along the path. Also get flow values
    for i in range(k-1):
        flows[path[i], path[i+1]] = 1
        flows[path[i+1], path[i]] = -1
        flowlist.append(capacaties[path[i],path[i+1]] + currentFlows[path[i+1],path[i]])

    # scale the flow directions by the minimum value and return it
    return flows * np.min(flowlist)


def solution(entrances, exits, path):
    """Find the maximum flow from the entrances to the exits. We will implement Dinic's algorithm for 
    optimal time complexity given a max flow of 2000000, and a max number of nodes of 50.
    inputs:
        entrances: list of entrance nodes - ndarray of shape (n,)
        exits: list of exit nodes - ndarray of shape (k,)
        path: path matrix - ndarray of shape (l, l)
        
    outputs:
        maxFlow: maximum flow from the entrances to the exits - int"""
    
    # Adjust the path matrix to include super source and super sink nodes to run Dinic's algorithm. Set iterate to True
    capacities = extend(entrances, exits, path)
    currentFlow = np.zeros_like(capacities)
    iterate = True

    # While there is a path from the super source to the super sink, run Dinic's algorithm
    while iterate:
        layers = bfs(capacities, currentFlow)

        # If there is no path from the super source to the super sink, set iterate to False. Otherwise, run the depth first search
        if layers[-1] == 0:
            iterate = False
        else:
            currentFlow = dfs(capacities, currentFlow, layers)
        
    # Return the maximum flow from the super source to the super sink
    return int(np.sum(currentFlow[:,-1]))


##################### Level 4, Problem 2 #####################
def solution(dimensions, your_position, trainer_position, distance):
    """Given the dimensions of a room, your position, the trainer's position, and the distance you can shoot, 
    return the number of times you can hit the trainer.
    inputs: 
        dimensions: x and y room dimensions - list of length 2
        your_position: x and y coordinates of your position - list of length 2
        trainer_position: x and y coordinates of the trainer's position - list of length 2
        distance: the distance you can shoot - int"""
    
    # Get the dimensions of the room as their own variables
    x_dim, y_dim = dimensions
    x_you, y_you = your_position
    x_target, y_target = trainer_position
    d2 = distance**2

    # Determine the distance until hitting your target in each direction and form the combinations of basis vectors
    x_right = x_target - x_you
    x_left =  -x_target - x_you
    y_up = y_target - y_you
    y_down = -y_target - y_you
    target_combos = [(x_right, y_up), (x_right, y_down), (x_left, y_up), (x_left, y_down)]

    # Determine the distance until hitting yourself in each direction and form the combinations of basis vectors
    x_you_right = 0
    x_you_left = -2*x_you
    y_you_up = 0
    y_you_down = -2*y_you
    you_combos = [(x_you_right, y_you_up), (x_you_right, y_you_down), (x_you_left, y_you_up), (x_you_left, y_you_down)]

    # Give an upper bound on the number of iterations needed to hit the target in each direction
    x_iterations = distance//(2 * x_dim) + 1
    y_iterations = distance//(2 * y_dim) + 1

    # Create a dictionary of confirmed hits on the youself using dictionary comprehension to improve time complexity. Also gets unique directions.
    hit_you = {(x//d,y//d):[]
                 for i in range(-x_iterations,x_iterations+1) for j in range(-y_iterations,y_iterations+1) 
                 for x_start,y_start in you_combos 
                 if (x := x_start + 2*x_dim*i) is not None 
                 if (y := y_start + 2*y_dim*j) is not None 
                 if x**2 + y**2 <= d2
                 if (d := abs(gcd(x,y))) != 0}                                   # Eliminate possible division by 0
    
    # Using the exact same code as above, just append the gcd to the list of hits. This will be used to determine the first hit
    Not_used = {hit_you[(x//d,y//d)].append(d) 
                 for i in range(-x_iterations,x_iterations+1) for j in range(-y_iterations,y_iterations+1) 
                 for x_start,y_start in you_combos 
                 if (x := x_start + 2*x_dim*i) is not None 
                 if (y := y_start + 2*y_dim*j) is not None 
                 if x**2 + y**2 <= d2
                 if (d := abs(gcd(x,y))) != 0}

    # Get the unique pairs of basis vectors that hit yourself first
    confirmed = set(hit_you.keys()) 

    # Using similar code as above, create a set of confirmed hits on the target 
    hit_target = {(x//d,y//d)
              for i in range(-x_iterations,x_iterations+1) for j in range(-y_iterations,y_iterations+1) 
              for x_start,y_start in target_combos
              if (x := x_start + 2*x_dim*i) is not None 
              if (y := y_start + 2*y_dim*j) is not None 
              if x**2 + y**2 <= d2
              if (d := abs(gcd(x,y))) != 0
              if (x//d,y//d) not in confirmed or min(hit_you[(x//d,y//d)]) > d}
    
    # Return the number of unique pairs of basis vectors that hit the target first
    return len(hit_target)