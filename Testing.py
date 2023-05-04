import numpy as np
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
    return newPath.tolist()


#build level graph by using BFS
def Bfs(C, F, s, t):  # C is the capacity matrix
        n = len(C)
        queue = []
        queue.append(s)
        global level
        level = n * [0]  # initialization
        level[s] = 1  
        while queue:
            k = queue.pop(0)
            for i in range(n):
                    if (F[k][i] < C[k][i]) and (level[i] == 0): # not visited
                            level[i] = level[k] + 1
                            queue.append(i)
        return level[t] > 0

#search augmenting path by using DFS
def Dfs(C, F, k, cp):
        tmp = cp
        if k == len(C)-1:
            return cp
        for i in range(len(C)):
            if (level[i] == level[k] + 1) and (F[k][i] < C[k][i]):
                f = Dfs(C,F,i,min(tmp,C[k][i] - F[k][i]))
                F[k][i] = F[k][i] + f
                F[i][k] = F[i][k] - f
                tmp = tmp - f
        return cp - tmp

#calculate max flow
#_ = float('inf')
def MaxFlow(C):
        s = 0
        t = len(C)-1
        n = len(C)
        F = [n*[0] for i in range(n)] # F is the flow matrix
        flow = 0
        while(Bfs(C,F,s,t)):
               flow = flow + Dfs(C,F,s,100000)
        return int(flow)



def solution1(entrances, exits, path):
    C = extend(entrances, exits, path)
    max_flow_value = MaxFlow(C)
    return max_flow_value


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
    global layers
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
    # 
    k = len(path)
    flows = np.zeros_like(capacaties)
    flowlist = []

    for i in range(k-1):
        flows[path[i], path[i+1]] = 1
        flows[path[i+1], path[i]] = -1
        flowlist.append(capacaties[path[i],path[i+1]] + currentFlows[path[i+1],path[i]])

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

    # While there is a path from the super source to the super sink, run Dinic's algorithm
    while True:
        layers = bfs(capacities, currentFlow)

        # If there is no path from the super source to the super sink, set iterate to False. Otherwise, run the depth first search
        if layers[-1] == 0:
            break
        else:
            currentFlow = dfs(capacities, currentFlow, layers)
        
    # Return the maximum flow from the super source to the super sink
    return int(np.sum(currentFlow[:,-1]))






# entrances, exits, path = [0, 1], [4, 5], [[0, 0, 4, 6, 0, 0], [0, 0, 5, 2, 0, 0], [0, 0, 0, 0, 4, 4], [0, 0, 0, 0, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
# entrances, exits, path = [0], [0], np.ones((50,50))*2000000


#make a 50 by 50 random matrix of integers
size = 1000
path = np.random.randint(0, 20, size=(size, size)) * np.random.randint(0, 2, size=(size, size)) * np.random.randint(0, 2, size=(size, size))
entrances = [0]
exits = [size-1]

import time
start = time.time()
solution1(entrances, exits, path)
end = time.time()
print("solution1 time:", end-start)

start = time.time()
solution(entrances, exits, path)
end = time.time()

print("solution time:", end-start)
print(path)


def tester(size):
    prevpath = 0
    while True:
        path = np.random.randint(0, 20, size=(size, size)) * np.random.randint(0, 2, size=(size, size)) * np.random.randint(0, 2, size=(size, size))
        entrances = [0]
        exits = [size-1]
        accurate = solution1(entrances, exits, path)

        approximate = solution(entrances, exits, path)
        if  accurate != approximate:
            break
        prevpath = path.copy()
        
    print(path)
    print(accurate)
    print(approximate)
    print(prevpath)

#tester(40)


# path1 = [[0, 15,  1,  0], [0, 0, 10, 0], [0, 0, 0, 9], [18, 0, 0,  0]]
# path2 = [[10,  9, 19,  7], [ 0,  0, 17,  8], [ 0, 10,  0, 18], [17,  4,  0,  0]]
# path3 = [[ 0, 16, 16, 14], [ 0,  0,  0, 17], [ 0, 10,  0, 11], [ 0,  0,  0,  0]]

# def testpath(path1):
#     print(np.array(path1))
#     print("Correct Solution \n")
#     print(solution1([0], [3], path1))
#     print("\nMy Solution \n")
#     print(solution([0], [3], path1))
#     #print(np.array(path1))

# testpath(path3)