import numpy as np
entrances, exits, path = [0, 1], [4, 5], [[0, 0, 4, 6, 0, 0], [0, 0, 5, 2, 0, 0], [0, 0, 0, 0, 4, 4], [0, 0, 0, 0, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]


def extend(entrances, exits, path):
    """Extend the path matrix by including super source and super sink nodes. 
    This will handle the case where there are multiple entrances and exits.
    inputs:
        entrances: list of entrance nodes - ndarray of shape (n,)
        exits: list of exit nodes - ndarray of shape (k,)
        path: path matrix - ndarray of shape (l, l)
        
    outputs:
        newPath: extended path matrix - ndarray of shape (l+2, l+2)"""
    # Get the number of nodes and number of additional nodes to be added. Initialize the new path matrix.
    n = len(path)
    newPath = np.zeros((n+2, n+2))

    # Fill in the new path matrix with the old path matrix and then add the super source and super sink nodes
    newPath[:-2, :-2] = path
    newPath[-1, entrances] = np.inf
    newPath[exits, -2] = np.inf
    
    # return the extended path matrix as a list of lists
    return newPath


def bfs(capacities, currentFlow):
    """Run a breadth first search to return the layer graph.
    """
    # Get number of nodes, initialize layers and layer graph. Set bottom layer to be the super sink nodes
    n = len(capacities)
    layers = np.zeros(n)

    # Get the first layer nodes and initialize the queue, queue nodes, and layers for the first layer
    firstLayer = np.where(capacities[-1] > 0)[0]
    queue = [(n-1,node) for node in firstLayer]
    queueNodes = [node for node in firstLayer]
    layers[firstLayer] = 1

    # While the queue is not empty, run the breadth first search
    while len(queue) != 0:
        # Pop the current node off from the queue and get the next steps
        prevNode, node = queue.pop()
        layers[node] = layers[prevNode] + 1
        nextStep = np.where((capacities[node] > 0) & (currentFlow[node] < capacities[node]))[0]

        # For each next step, if the node is not backtracking, fill in the layer graph
        for test in nextStep:

            # if the node has not been visited, add it to the queue and queue nodes
            if test not in queueNodes:
                queue.insert(0,(node,test)) 
                queueNodes.append(test)
    
    # Return the layer nodes
    return layers

#Start from the back and use the level graph to get the path to the super source. Then find the values where the left is 1 less than the right. Take those values in a list. Index the queue using those values 
def dfs(capacities, currentFlow, layers):
    """Run a depth first search to return the blocking flow.
    """
    n = len(capacities)
    traverse = capacities > 0
    queue = [n-1]
    levelPath = [layers[n-1]]
    flows = []
    path = []
    dummy = [n-1]
    truepath = []
    
    while len(queue) != 0:
        node = queue.pop()
        
        if node == n-2:
            dummy = dummy[::-1]
            levelPath = layers[dummy]
            flows = flows[::-1]
            curMin = np.inf
            for i,level in enumerate(levelPath[:-1]):
                if level < curMin:
                    path.append(flows[i])
                    truepath.append(dummy[i])
                    curMin = level
            break

        nextStep = np.where(traverse[node] == True)[0]
        for test in nextStep:

            if (layers[test] == layers[node] + 1) and test not in queue:
                queue.append(test)
                flows.append(capacities[node, test])
                dummy.append(test)
                traverse[node, test] = False

    update = np.min(path)

    for i in range(len(truepath)):
        current
    print(currentFlow)

    return path, truepath
        
        
                

path = np.array([[0,2,2,0,0,0],[0,0,0,0,0,0],[0,0,0,1,3,0],[0,0,0,0,0,0],[0,0,0,0,0,4],[0,0,0,0,0,0]])
entrances = [0]
exits = [5]
extended = extend(entrances, exits, path)
currentFlow = np.zeros_like(extended)
layers = bfs(extended, currentFlow)
#dfs(extended, currentFlow, layerGraph, max
#print(layers)
print(dfs(extended, currentFlow, layers))




def solution(entrances, exits, path):
    """Find the maximum flow from the entrances to the exits. We will implement Dinic's algorithm for 
    optimal time complexity given a max flow of 2000000, and a max number of nodes of 50.
    inputs:
        entrances: list of entrance nodes - ndarray of shape (n,)
        exits: list of exit nodes - ndarray of shape (k,)
        path: path matrix - ndarray of shape (l, l)
        
    outputs:
        maxFlow: maximum flow from the entrances to the exits - int"""
    
    # Adjust the path matrix to include super source and super sink nodes to run Dinic's algorithm
    capacities = extend(entrances, exits, path)
    currentFlow = np.zeros_like(capacities)
    maxFlow = 0

    # While there is a path from the super source to the super sink, run Dinic's algorithm
    while bfs(capacities, currentFlow):
        break

# remaining capacity >0 `and` current flow < capacity