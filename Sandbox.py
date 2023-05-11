import numpy as np
from math import gcd
from time import time

def solution(dimensions, your_position, trainer_position, distance):
    x_dim, y_dim = dimensions
    x_you, y_you = your_position
    x_target, y_target = trainer_position
    d2 = distance**2

    if x_you <= x_target:
        x_right = x_target - x_you
        x_left =  -x_target - x_you
    else:
        x_right = 2*x_dim - x_you - x_target
        x_left = x_target - x_you

    if y_you <= y_target:
        y_up = y_target - y_you
        y_down = -y_target - y_you
    else:
        y_up = 2*y_dim - y_you - y_target
        y_down = y_target - y_you

    x_iterations = distance//(2 * x_dim) + 1
    y_iterations = distance//(2 * y_dim) + 1

    target_combos = [(x_right, y_up), (x_right, y_down), (x_left, y_up), (x_left, y_down)]
    hit_target = {(x//d,y//d) for i in range(-x_iterations,x_iterations+1) for j in range(-y_iterations,y_iterations+1) 
              for x_start,y_start in target_combos 
              if (x := x_start + 2*x_dim*i) is not None 
              if (y := y_start + 2*y_dim*j) is not None 
              if y**2 + x**2 <= d2 
              if (d := gcd(x,y))!=0}

    # right_up = {(x//d,y//d) for i in range(-iterations,iterations+1) for j in range(-iterations,iterations+1) if (x := x_right + 2*x_dim*i) is not None if (y := y_up + 2*y_dim*j) is not None if y**2 + x**2 <= d2 if (d := gcd(x,y))!=0}
    # right_down = {(x//d,y//d) for i in range(-iterations,iterations+1) for j in range(-iterations,iterations+1) if (x := x_right + 2*x_dim*i) is not None if (y := y_down + 2*y_dim*j) is not None if y**2 + x**2 <= d2 if (d := gcd(x,y))!=0}
    # left_up = {(x//d,y//d) for i in range(-iterations,iterations+1) for j in range(-iterations,iterations+1) if (x := x_left + 2*x_dim*i) is not None if (y := y_up + 2*y_dim*j) is not None if y**2 + x**2 <= d2 if (d := gcd(x,y))!=0}
    # left_down = {(x//d,y//d) for i in range(-iterations,iterations+1) for j in range(-iterations,iterations+1) if (x := x_left + 2*x_dim*i) is not None if (y := y_down + 2*y_dim*j) is not None if y**2 + x**2 <= d2 if (d := gcd(x,y))!=0}

    # target = right_up.union(right_down).union(left_up).union(left_down)
    # print(target)
    x_you_right = 0
    x_you_left = -2*x_you
    y_you_up = 0
    y_you_down = -2*y_you
    confirmed_hits = {}
    you_combos = [(x_you_right, y_you_up), (x_you_right, y_you_down), (x_you_left, y_you_up), (x_you_left, y_you_down)]

    hit_you = {(x//d,y//d) for i in range(-x_iterations,x_iterations+1) for j in range(-y_iterations,y_iterations+1) 
              for x_start,y_start in you_combos 
              if (x := x_start + 2*x_dim*i) is not None 
              if (y := y_start + 2*y_dim*j) is not None 
              if y**2 + x**2 <= d2 
              if (d := gcd(x,y))!=0}
  
    #hit_you = hit_you.union({(0,1), (0,-1), (1,0), (-1,0)})
    
    # num_target_hits = len(hit_target)
    # num_you_hits = len(hit_you)
    # target_first = 0

    # # Loop through the intersection of the two sets
    # for x, y in hit_you.intersection(hit_target):
    #     for i in range(1,distance+1):
            
            
    #         target_first += 1 
    
    # return num_target_hits - num_you_hits + target_first
    #print(hit_target)
    return hit_you

print(solution([3, 2], [1, 1], [2, 1], 7))

yaboi = {(1,2):3}



