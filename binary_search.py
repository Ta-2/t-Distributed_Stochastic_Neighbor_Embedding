import numpy as np

def binary_search(func, args, target, threshold, search_range, loop_limit=100):
    cnt = 0
    #print("target: " + str(target))
    L_lim = func(search_range[0], *args)
    #print("L_lim: " + str(L_lim))
    if L_lim > target:
        print("target value is less than smallest of range")
        return search_range[0]
    R_lim = func(search_range[1], *args)
    #print("R_lim: " + str(R_lim))
    if R_lim < target:
        print("target value is more than biggest of range")
        return search_range[1]
    
    while(True):
        #print("cnt: " + str(cnt))

        L, R = search_range[0], search_range[1]
        C = (L + R)/2.0
        #print("C: " + str(C))

        f_result = func(C, *args)
        #print("f_result: " + str(f_result))

        if(f_result < target):
            search_range = [C, R]
        else:
            search_range = [L, C]
        #print("search_range: " + str(search_range))

        cnt += 1
        #print()

        if cnt>loop_limit or abs(target - f_result) < threshold:
            f_result = func(C, *args)
            #print("f_result: " + str(f_result))
            return C