# The skipping model
# @author Rosa Zhou
# @author Will Thompson

from cost_model import cost_model
import random
import math
from input_reader import *
import matplotlib.pyplot as plt
import scipy.stats
from global_skipping import *
import time

def random_simulation(L, J, I0, h, a, trigger_points, D, t, Tau, T, num_simulation, optimal_lambda, neighbourhood):
    '''
    This function runs a random simulation to test different combinations of
    Lambdas

    PARAM:
    L: number of items
    J: number of time periods
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: A list of lists containing all item demands in each time period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    num_simulation: the number of simulations to run
    optimal_lambda: the output from the Cost Model that optimizes Base Loop
                    without skipping
    neighbourhood: the interval around each lambda that we will sample new choices
                   of lambda from

    RETURN:
    A dictionary containing feasible choices for the lambdas and their respective
    average Base Loop times
    '''
    feasible_results = {}
    for i in range(num_simulation):
        Lambda = get_random_lambdas(optimal_lambda, neighbourhood)
        avg_baseloop = get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, False)

        if avg_baseloop != -1:
            feasible_results[avg_baseloop] = Lambda

    print('the size of the dictionary is {}'.format(len(feasible_results)))
    return feasible_results


def get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info):
    '''
    This function loops through each time period and checks the skipping criteria,


    PARAM:
    L: number of items
    J: number of time periods
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: A list of lists containing all item demands in each time period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    print_optimal_info: a boolean that allows you to print out additional
                        information about cost

    RETURN:
    Average Base Loop time
    '''
    inventory = []

    # initialize placeholders (all zeros) for skipping coefficients
    S = []
    for time_index in range(J):
        S.append([0] * L)

    # initialization
    cur_inventory = I0.copy()
    total_baseloop = 0
    total_holding_cost = 0
    total_changeover_cost = 0

    for j in range(J):

        inventory_j = []
        # determine which items to skip
        for i in range(L):
            item_inventory = cur_inventory[i]
            if item_inventory < max(trigger_points[i], D[j][i]):
                # produce this month
                S[j][i] = 1
        # compute baseloop at time j

        baseloop = get_baseloop_skipping(Lambda, t, S[j])
        total_baseloop += baseloop
        for i in range(L):
            # feasibility: meet demand at each time period
            if S[j][i] == 1:

                # number of base loop
                num_baseloop = math.floor(T / baseloop)

                production = Lambda[i] * num_baseloop

                #print('num_baseloop is{}, production is{}'.format(num_baseloop,production))

                # There is only 1 or 0 item, then there is no changeover cost
                if sum([coeff for coeff in S[j]]) > 1:
                    total_changeover_cost += a[i] * num_baseloop
                if production + cur_inventory[i] < D[j][i]:

                    # does not meet demand
                    if print_optimal_info: print('Does not meet demand')
                    return -1
            else:
                production = 0

            inventory_j.append(production + cur_inventory[i]- D[j][i])
            #Note I added thi ssubtraction
            # update inventory
            # Not sure about this total holding cost whether it accounts for the leftover or the real current
            cur_inventory[i] = production + cur_inventory[i] - D[j][i]
            # update holding cost
            total_holding_cost += h[i] * cur_inventory[i]
        inventory.append(inventory_j)

    # feasibility: cost tolerance in a year
    if total_holding_cost + total_changeover_cost > Tau:
        if print_optimal_info: print('Exceeds cost tolerance')
        return -1

    avg_baseloop = total_baseloop / (J)
    if print_optimal_info:
        print('average baseloop time is: ', avg_baseloop)
        print('skipping coefficients: ', S)
        print('inventory: ', inventory)
        print('total_holding_cost: ', total_holding_cost)
        print('total_changeover_cost: ', total_changeover_cost)
        print('total_cost,',total_changeover_cost+total_holding_cost)

    return avg_baseloop

def get_greedy_and_potential(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info):
    '''
    This function loops through each time period and checks the skipping criteria,
    PARAM:
    L: number of items
    J: number of time periods
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: A list of lists containing all item demands in each time period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    print_optimal_info: a boolean that allows you to print out additional
                        information about cost

    RETURN:
    Average Base Loop time
    '''
    inventory = []

    # initialize placeholders (all zeros) for skipping coefficients
    S = []
    for time_index in range(J):
        S.append([0] * L)

    # initialization
    cur_inventory = I0.copy()
    total_baseloop = 0
    total_holding_cost = 0
    total_changeover_cost = 0
    total_baseloop_in_the_first=0

    for j in range(J):
        inventory_j = []
        # determine which items to skip
        for i in range(L):
            item_inventory = cur_inventory[i]
            if item_inventory < max(trigger_points[i], D[j][i]):
                # produce this month
                S[j][i] = 1
        # compute baseloop at time j

        baseloop = get_baseloop_skipping(Lambda, t, S[j])
        total_baseloop += baseloop


        if j==0:
            total_baseloop_in_the_first+=baseloop

        for i in range(L):
            # feasibility: meet demand at each time period
            if S[j][i] == 1:

                # number of base loop
                num_baseloop = math.floor(T / baseloop)

                production = Lambda[i] * num_baseloop

                #print('num_baseloop is{}, production is{}'.format(num_baseloop,production))

                # There is only 1 or 0 item, then there is no changeover cost
                if sum([coeff for coeff in S[j]]) > 1:
                    total_changeover_cost += a[i] * num_baseloop
                if production + cur_inventory[i] < D[j][i]:
                    # does not meet demand
                    if print_optimal_info: print('Does not meet demand')
                    return -1, None, None
            else:
                production = 0

            inventory_j.append(production + cur_inventory[i]- D[j][i])
            # update inventory
            # Not sure about this total holding cost whether it accounts for the leftover or the real current
            cur_inventory[i] = production + cur_inventory[i] - D[j][i]
            # update holding cost
            total_holding_cost += h[i] * cur_inventory[i]
        inventory.append(inventory_j)

    # feasibility: cost tolerance in a year
    if total_holding_cost + total_changeover_cost > Tau:
        if print_optimal_info: print('Exceeds cost tolerance')
        return -1, None, None

    S_potential=np.array(S)
    S_potential=np.delete(S_potential,0,0)
    S_indicate=np.sum(S_potential,axis=0)
    S_indicate=np.multiply(S_indicate>0,1)
    S_indicate=S_indicate.tolist()

    potential=total_baseloop_in_the_first+get_baseloop_skipping(Lambda, t, S_indicate)



    if print_optimal_info:
        print('total_baseloop time is: ', total_baseloop)
        print('skipping coefficients: ', S)
        print('inventory: ', inventory)
        print('total_holding_cost: ', total_holding_cost)
        print('total_changeover_cost: ', total_changeover_cost)
        print('total_cost,',total_changeover_cost+total_holding_cost)

    return total_baseloop, S,potential


def get_cost(S,L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info):
    '''
    This function loops through each time period and checks the skipping criteria,
    PARAM:
    L: number of items
    J: number of time periods
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: A list of lists containing all item demands in each time period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    print_optimal_info: a boolean that allows you to print out additional
                        information about cost

    RETURN:
    Average Base Loop time
    '''
    inventory = []

    # initialize placeholders (all zeros) for skipping coefficients


    # initialization
    cur_inventory = I0.copy()
    total_baseloop = 0
    total_holding_cost = 0
    total_changeover_cost = 0
    total_baseloop_in_the_first=0

    for j in range(J):
        inventory_j = []
        # determine which items to skip


        baseloop = get_baseloop_skipping(Lambda, t, S[j])
        total_baseloop += baseloop


        if j==0:
            total_baseloop_in_the_first+=baseloop

        for i in range(L):
            # feasibility: meet demand at each time period
            if S[j][i] == 1:

                # number of base loop
                num_baseloop = math.floor(T / baseloop)

                production = Lambda[i] * num_baseloop

                #print('num_baseloop is{}, production is{}'.format(num_baseloop,production))

                # There is only 1 or 0 item, then there is no changeover cost
                if sum([coeff for coeff in S[j]]) > 1:
                    total_changeover_cost += a[i] * num_baseloop
                if production + cur_inventory[i] < D[j][i]:
                    # does not meet demand
                    if print_optimal_info: print('Does not meet demand')
                    return -1, None, None
            else:
                production = 0

            inventory_j.append(production + cur_inventory[i]- D[j][i])
            # update inventory
            # Not sure about this total holding cost whether it accounts for the leftover or the real current
            cur_inventory[i] = production + cur_inventory[i] - D[j][i]
            # update holding cost
            total_holding_cost += h[i] * cur_inventory[i]
        inventory.append(inventory_j)

    # feasibility: cost tolerance in a year
    if total_holding_cost + total_changeover_cost > Tau:
        if print_optimal_info: print('Exceeds cost tolerance')
        return -1, None, None

    S_potential=np.array(S)
    S_potential=np.delete(S_potential,0,0)
    S_indicate=np.sum(S_potential,axis=0)
    S_indicate=np.multiply(S_indicate>0,1)
    S_indicate=S_indicate.tolist()

    potential=total_baseloop_in_the_first+get_baseloop_skipping(Lambda, t, S_indicate)



    if print_optimal_info:
        print('total_baseloop time is: ', total_baseloop)
        print('skipping coefficients: ', S)
        print('inventory: ', inventory)
        print('total_holding_cost: ', total_holding_cost)
        print('total_changeover_cost: ', total_changeover_cost)
        print('total_cost,',total_changeover_cost+total_holding_cost)

    return total_baseloop, S,potential



def get_candidate(L,  I0, h, a, D, Lambda, t, T):
    '''
    This function finds all candidates under demand constraints in a given period


    PARAM:
    L: number of items
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: Demand in one period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    print_optimal_info: a boolean that allows you to print out additional
                        information about cost

    RETURN:
    candidate inventories and skippings
    '''


    candidate_skipping=[]
    candidate_inventory=[]
    candidate_changeover_cost=[]
    candidate_inventory_cost=[]
    candidate_baseloop=[]

    # initialize placeholders (all zeros) for skipping coefficients
    S = [0]*L

    for i in range(L):
        if I0[i] <  D[i]:
            # produce this month
            S[i] = 1

    S_np=np.array(S)
    num_cadi=2**(L-sum(S))
    index_choice=np.where(S_np==0)[0]


    for i in range(num_cadi):
        new_index=np.array(list(np.binary_repr(i).zfill(L-sum(S)))).astype(np.int8)
        S_cur_np=np.copy(S_np)

        S_cur_np[index_choice]=new_index
        #new candidate
        S_array=S_cur_np.tolist()

        new_inventory=[]
        total_inventory_cost=0
        total_changeover=0



        for j in range(L):
            # feasibility: meet demand at each time period

            if S_array[j] == 1:
                baseloop = get_baseloop_skipping(Lambda, t, S_array)
                # number of base loop
                num_baseloop = math.floor(T / baseloop)

                production = Lambda[j] * num_baseloop

                if sum(S_array) > 1:
                    total_changeover += a[j] * num_baseloop

                if I0[j]+production-D[j]<0:
                    break

            else:
                production=0

            left_j=I0[j]+production-D[j]
            new_inventory.append(left_j)
            total_inventory_cost += h[j] * left_j

            if j==L-1:

                total_baseloop = get_baseloop_skipping(Lambda, t, S_array)

                candidate_skipping.append(S_array)
                candidate_inventory.append(new_inventory)
                candidate_changeover_cost.append(total_changeover)
                candidate_inventory_cost.append(total_inventory_cost)
                candidate_baseloop.append(total_baseloop)


    #print('candidate_skipping is {}'.format(candidate_skipping))
    #print('candidate_inventory is {}'.format(candidate_inventory))
    #print('candidate_changeover_cost is {}'.format(candidate_changeover_cost))
    #print('candidate_inventory_cost is {}'.format(candidate_inventory_cost))
    #print('candidate_baseloop is {}'.format(candidate_baseloop))
    return candidate_skipping, candidate_inventory,candidate_changeover_cost,candidate_inventory_cost,candidate_baseloop

def get_candidate_for_first_level(L,  I0, h, a, D, Lambda, t, T):
    '''
    This function finds all candidates under demand constraints in a given period


    PARAM:
    L: number of items
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: Demand in one period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    print_optimal_info: a boolean that allows you to print out additional
                        information about cost

    RETURN:
    candidate inventories and skippings
    '''


    candidate_skipping=[]
    candidate_inventory=[]
    candidate_changeover_cost=[]
    candidate_inventory_cost=[]
    candidate_baseloop=[]

    # initialize placeholders (all zeros) for skipping coefficients
    S = [0]*L

    for i in range(L):
        if I0[i] <  D[i]:
            # produce this month
            S[i] = 1

    S_np=np.array(S)
    num_cadi=2**(L-sum(S))
    index_choice=np.where(S_np==0)[0]


    for i in range(num_cadi):
        new_index=np.array(list(np.binary_repr(i).zfill(L-sum(S)))).astype(np.int8)
        S_cur_np=np.copy(S_np)

        S_cur_np[index_choice]=new_index
        #new candidate
        S_array=S_cur_np.tolist()

        new_inventory=[]
        total_inventory_cost=0
        total_changeover=0
        total_baseloop=0


        for j in range(L):
            # feasibility: meet demand at each time period

            if S_array[j] == 1:
                baseloop = get_baseloop_skipping(Lambda, t, S_array)
                # number of base loop
                num_baseloop = math.floor(T / baseloop)

                production = Lambda[j] * num_baseloop

                if sum(S_array) > 1:
                    total_changeover += a[j] * num_baseloop

                if I0[j]+production-D[j]<0:
                    break


            else:
                production=0

            left_j=I0[j]+production-D[j]
            new_inventory.append(left_j)
            total_inventory_cost += h[j] * left_j

            if j==L-1:

                total_baseloop = get_baseloop_skipping(Lambda, t, S_array)

                candidate_skipping.append([S_array])
                candidate_inventory.append(new_inventory)
                candidate_changeover_cost.append(total_changeover)
                candidate_inventory_cost.append(total_inventory_cost)
                candidate_baseloop.append(total_baseloop)


    return candidate_skipping, candidate_inventory,candidate_changeover_cost,candidate_inventory_cost,candidate_baseloop

def explore_fully(level,L, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info):
    # how many periods you would like to get candidate
    if level==1:

        candidate_skipping, candidate_inventory,candidate_changeover_cost,candidate_inventory_cost,candidate_baseloop = get_candidate_for_first_level(L,  I0, h, a, D[0], Lambda, t, T)
        #print('this is first level')

        return candidate_skipping, candidate_inventory,candidate_changeover_cost,candidate_inventory_cost,candidate_baseloop
    else:
        p_candidate_skipping, p_candidate_inventory,p_candidate_changeover_cost,p_candidate_inventory_cost,p_candidate_baseloop=explore_fully(level-1,L, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info)

        n=len(p_candidate_skipping)

        c_candidate_skipping=[]
        c_candidate_inventory=[]
        c_candidate_changeover_cost=[]
        c_candidate_inventory_cost=[]
        c_candidate_baseloop=[]

        for i in range(n):
            n_candidate_skipping, n_candidate_inventory, n_candidate_changeover_cost, n_candidate_inventory_cost, n_candidate_baseloop=get_candidate(L, p_candidate_inventory[i], h, a, D[level-1], Lambda, t, T)


            m=len(n_candidate_skipping)
            for j in range(m):
                new_total_inventory_cost=p_candidate_inventory_cost[i]+n_candidate_inventory_cost[j]
                new_total_changeover_cost =p_candidate_changeover_cost[i]+n_candidate_changeover_cost[j]
                new_total_cost=new_total_inventory_cost+new_total_changeover_cost

                if new_total_cost<=Tau:



                    new_baseloop=p_candidate_baseloop[i]+n_candidate_baseloop[j]

                    new_skipping=list.copy(p_candidate_skipping[i])
                    new_skipping.append(n_candidate_skipping[j])



                    c_candidate_skipping.append(new_skipping)
                    c_candidate_inventory.append(n_candidate_inventory[j])
                    c_candidate_changeover_cost.append(new_total_changeover_cost)
                    c_candidate_inventory_cost.append(new_total_inventory_cost)
                    c_candidate_baseloop.append(new_baseloop)
        #print('level is {}'.format(level))
        #print(c_candidate_skipping, c_candidate_inventory,c_candidate_changeover_cost,c_candidate_inventory_cost,c_candidate_baseloop)
        return c_candidate_skipping, c_candidate_inventory,c_candidate_changeover_cost,c_candidate_inventory_cost,c_candidate_baseloop


def explore_next_level_and_bound(c_candidate_skipping, c_candidate_inventory,\
                                 c_candidate_changeover_cost,c_candidate_inventory_cost,\
                                 c_candidate_baseloop,level,bound_skipping,bound,t, Lambda, T,demand,L,h,a,Tau,trigger_points,J):

    # level is the number of period that we already produced
    # THis is all demand


    #print('this is previous candidate {}'.format(c_candidate_skipping))

    #print('this is previous candidate inventory {}'.format(c_candidate_inventory))
    N=len(c_candidate_skipping)

    n_candidate_skipping=[]
    n_candidate_inventory=[]
    n_candidate_changeover_cost=[]
    n_candidate_inventory_cost=[]
    n_candidate_baseloop=[]
    n_candidate_lower_b_baseloop=[]
    min_check=bound
    min_skipping=bound_skipping

    #print('level is {}'.format(level))
    #print('N is {}'.format(N))
    for i in range(N):

        a_candidate_skipping, a_candidate_inventory, a_candidate_changeover_cost,\
        a_candidate_inventory_cost, a_candidate_baseloop = get_candidate(L, c_candidate_inventory[i], \
                                                                         h, a, demand[level], Lambda, t, T)


        #print("candidate inventory after that{}".format(a_candidate_inventory))
        #print("candidate after that {}".format(a_candidate_skipping))
        m = len(a_candidate_skipping)
        #print(' m is {}'.format(m))
        for j in range(m):

            total_baseloop_added, S_added, potential_added=get_greedy_and_potential(L, J-level-1, a_candidate_inventory[j], h, a, trigger_points, \
                                                                                    demand[level+1:J], Lambda, t,\
                                                                                    Tau-a_candidate_changeover_cost[j]-\
                                                                                    a_candidate_inventory_cost[j]-c_candidate_changeover_cost[i]-c_candidate_inventory_cost[i], T, False)
            if not total_baseloop_added==-1:

                current_candidate_skipping=list.copy(c_candidate_skipping[i])
                current_candidate_skipping.append(a_candidate_skipping[j])

                current_baseloop=a_candidate_baseloop[j]+c_candidate_baseloop[i]

                current_changeover_cost=c_candidate_changeover_cost[i]+a_candidate_changeover_cost[j]
                current_inventory_cost=c_candidate_inventory_cost[i]+a_candidate_inventory_cost[j]
                current_inventory=list.copy(a_candidate_inventory[j])

                #print('is this sopt')
                #print(a_candidate_baseloop[j])
                #print(total_baseloop_added)
                #print(c_candidate_baseloop[i])
                #print(min_check)
                greedy_full_baseloop=a_candidate_baseloop[j]+total_baseloop_added+c_candidate_baseloop[i]

                greedy_full_baseloop_skipping=current_candidate_skipping+S_added
                lower_b_baseloop=a_candidate_baseloop[j]+potential_added+c_candidate_baseloop[i]


                if greedy_full_baseloop<min_check:
                    min_check=greedy_full_baseloop
                    min_skipping=greedy_full_baseloop_skipping

                if lower_b_baseloop < bound:

                    n_candidate_skipping.append(current_candidate_skipping)
                    n_candidate_inventory.append(current_inventory)
                    n_candidate_changeover_cost.append(current_changeover_cost)
                    n_candidate_inventory_cost.append(current_inventory_cost)
                    n_candidate_baseloop.append(current_baseloop)
                    n_candidate_lower_b_baseloop.append(lower_b_baseloop)



    r_candidate_skipping=[]
    r_candidate_inventory=[]
    r_candidate_changeover_cost=[]
    r_candidate_inventory_cost=[]
    r_candidate_baseloop=[]
    K=len(n_candidate_skipping)



    for i in range(K):
        if  n_candidate_lower_b_baseloop[i]<min_check:
            r_candidate_skipping.append(n_candidate_skipping[i])
            r_candidate_inventory.append(n_candidate_inventory[i])
            r_candidate_changeover_cost.append(n_candidate_changeover_cost[i])
            r_candidate_inventory_cost.append(n_candidate_inventory_cost[i])
            r_candidate_baseloop.append(n_candidate_baseloop[i])




    return r_candidate_skipping, r_candidate_inventory,r_candidate_changeover_cost,r_candidate_inventory_cost,r_candidate_baseloop, min_check,min_skipping


def get_global(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info):

    level=1

    total_baseloop_greedy, S_greedy, potential_added= get_greedy_and_potential(L, J ,I0, h, a,trigger_points,D, Lambda, t, Tau , T, False)


    c_candidate_skipping, c_candidate_inventory,c_candidate_changeover_cost,c_candidate_inventory_cost,c_candidate_baseloop = explore_fully(level,L, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info)

    #print('check inventory from fully {}'.format(c_candidate_inventory))

    #print('len of c {}'.format(len(c_candidate_skipping)))

    #print("this is from greedy {} and {}".format(total_baseloop_greedy,S_greedy))
    #print('this is also from greedy {}'.format(c_candidate_skipping))
    #print('this is also from greedy {}'.format(c_candidate_inventory))

    min_baseloop_g=total_baseloop_greedy
    min_skipping_g=S_greedy


    for i in range(level,10):

        print('This is real level {}'.format(i))
        print('this is step {}'.format(i))
        r_candidate_skipping, r_candidate_inventory,\
        r_candidate_changeover_cost,r_candidate_inventory_cost,\
        r_candidate_baseloop, min_check,min_skipping=explore_next_level_and_bound(c_candidate_skipping, c_candidate_inventory, c_candidate_changeover_cost,
                                     c_candidate_inventory_cost, c_candidate_baseloop, i, min_skipping_g, min_baseloop_g, t,
                                     Lambda, T, D, L, h, a, Tau, trigger_points, J)

        #print('this is candidate skipping {}'.format(r_candidate_skipping))

        if len(r_candidate_skipping)==0:
            print('The global optimum is found')
            print('The baseloop is {}. And the skipping coef is {}'.format(min_baseloop_g,min_skipping_g))

            return min_baseloop_g,min_skipping_g

        else:
            min_baseloop_g=min_check
            min_skipping_g=min_skipping

        c_candidate_skipping=list.copy(r_candidate_skipping)
        c_candidate_inventory=list.copy(r_candidate_inventory)
        c_candidate_changeover_cost=list.copy(r_candidate_changeover_cost)
        c_candidate_inventory_cost=list.copy(r_candidate_inventory_cost)
        c_candidate_baseloop=list.copy(r_candidate_baseloop)


def get_neighbor_global(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info, neighborhood_sd,iteration):


    surveyed_skipping={}
    surveyed_baseloop = {}

    min_baseloop_given, min_skipping_given = get_global(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T,
                                                print_optimal_info)


    min_key_f=tuple(Lambda)
    surveyed_skipping[min_key_f]=min_skipping_given
    surveyed_baseloop[min_key_f] = min_baseloop_given


    for i in range(iteration):
        new_lambda=get_random_lambdas_mcmc_random(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info,neighborhood_sd)

        new_lambda_t=tuple(new_lambda)
        if new_lambda_t in surveyed_baseloop:
            min_baseloop_b=surveyed_baseloop[new_lambda_t]
            min_skipping_b=surveyed_skipping[new_lambda_t]
            print('{} was in dict'.format(i))

        else:

            min_baseloop_b, min_skipping_b=get_global(L, J, I0, h, a, trigger_points, D, new_lambda, t, Tau, T, print_optimal_info)
            surveyed_skipping[new_lambda_t] = min_skipping_b
            surveyed_baseloop[new_lambda_t] = min_baseloop_b




        if min_baseloop_given>min_baseloop_b:
            min_baseloop_given=min_baseloop_b
            min_skipping_given=min_skipping_b


    print('the global around here is {} by {}'.format(min_baseloop_given,min_skipping_given))
    return min_baseloop_given,min_skipping_given



def get_average_baseloop_time_compare(L, J, I0, h, a, trigger_points, D, Lambda, t, Tau, T, print_optimal_info):
    inventory = []

    # initialize placeholders (all zeros) for skipping coefficients
    S = []
    for time_index in range(J):
        S.append([0] * L)

    # initialization
    cur_inventory = I0.copy()
    total_baseloop = 0
    total_holding_cost = 0
    total_changeover_cost = 0

    for j in range(J):
        inventory_j = []
        # determine which items to skip
        for i in range(L):
            item_inventory = cur_inventory[i]
            if item_inventory < max(trigger_points[i], D[j][i]):
                # produce this month
                S[j][i] = 1
        # compute baseloop at time j

        baseloop = get_baseloop_skipping(Lambda, t, S[j])
        total_baseloop += baseloop
        for i in range(L):
            # feasibility: meet demand at each time period
            if S[j][i] == 1:

                # number of base loop
                num_baseloop = math.floor(T / baseloop)

                production = Lambda[i] * num_baseloop

                #print('num_baseloop is{}, production is{}'.format(num_baseloop,production))

                # There is only 1 or 0 item, then there is no changeover cost
                if sum([coeff for coeff in S[j]]) > 1:
                    total_changeover_cost += a[i] * num_baseloop
                if production + cur_inventory[i] < D[j][i]:
                    # does not meet demand
                    if print_optimal_info: print('Does not meet demand')
                    return -1
            else:
                production = 0

            inventory_j.append(production + cur_inventory[i])
            # update inventory
            # Not sure about this total holding cost whether it accounts for the leftover or the real current
            cur_inventory[i] = production + cur_inventory[i] - D[j][i]
            # update holding cost
            total_holding_cost += h[i] * cur_inventory[i]
        inventory.append(inventory_j)

    # feasibility: cost tolerance in a year
    if total_holding_cost + total_changeover_cost > Tau:
        if print_optimal_info: print('Exceeds cost tolerance')
        return -1

    avg_baseloop = total_baseloop / (J)
    if print_optimal_info:
        print('average bas'
              'eloop time is: ', avg_baseloop)
        print('skipping coefficients: ', S)
        print('inventory: ', inventory)
        print('total_holding_cost: ', total_holding_cost)
        print('total_changeover_cost: ', total_changeover_cost)
        print('total_cost,',total_changeover_cost+total_holding_cost)

    return avg_baseloop, total_changeover_cost+total_holding_cost

def get_baseloop_skipping(Lambda, t, s):
    '''
    This function computes the baseloop at a given time period

    PARAM:
    Lambda: a list of L items, each correspond to number of one item
    produced in a loop
    t: a list of time takes to produce one unit of item
    s: a list of L skipping coeffs for this time period

    RETURN:
    Base Loop time
    '''
    baseloop = 0
    for i in range(len(Lambda)):
        baseloop += Lambda[i] * t[i] * s[i]
        #print('baseloop is{}'.format(baseloop))


    return baseloop

def get_random_lambdas(optimal_lambda, neighborhood):
    '''
    This function randomly samples from an interval around each lambda

    PARAM:
    neighbourhood: the interval around each lambda that we will sample new choices
                   of lambda from
    optimal_lambda: a list of L items output by the non-skipping model

    RETURN:
    A new choice of lambdas
    '''

    #comment this neightbor should be matrix
    new_lambda = optimal_lambda.copy()
    for i in range(len(optimal_lambda)):
        generated_val = -1
        while generated_val <= 0:
            generated_val = (random.uniform(optimal_lambda[i] - neighborhood, \
                                               optimal_lambda[i] + neighborhood))
        new_lambda[i] = generated_val
    return new_lambda

def get_random_lambdas_normal(optimal_lambda, neighborhood_sd):
    '''
    This function randomly samples from an interval around each lambda

    PARAM:
    neighbourhood: the interval around each lambda that we will sample new choices
                   of lambda from
    optimal_lambda: a list of L items output by the non-skipping model

    RETURN:
    A new choice of lambdas
    '''

    #comment this neightbor should be matrix
    new_lambda = optimal_lambda.copy()
    for i in range(len(optimal_lambda)):
        generated_val = -1
        while generated_val <= 0:
            generated_val = np.random.normal(optimal_lambda[i],neighborhood_sd)
        new_lambda[i] = generated_val
    return new_lambda

def get_random_lambdas_mcmc(L, J, I0, h, a, trigger_points, D, optimal_lambda, t, Tau, T, print_optimal_info,neighborhood):


    new_lambda = get_random_lambdas(optimal_lambda, neighborhood)

    while(get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, new_lambda, t, Tau, T, False)==-1):
        new_lambda = get_random_lambdas(optimal_lambda, neighborhood)


    return new_lambda


def get_random_lambdas_mcmc_random(L, J, I0, h, a, trigger_points, D, optimal_lambda, t, Tau, T, print_optimal_info,neighborhood_sd):
    new_lambda = get_random_lambdas_normal(optimal_lambda, neighborhood_sd)

    while( get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, new_lambda, t, Tau, T, False) == -1 ):
        new_lambda = get_random_lambdas_normal(optimal_lambda, neighborhood_sd)
    return new_lambda


def get_optimal_siumulation_results(some_simulation_result):
    '''
    This function takes some output from the simulation, and loops through the
    results and finds which combination of lambdas produced the smallest
    average Base Loop

    PARAMETERS:
    some_simulation_result := Some dictionary of feasible results outputed from
                              the random simulation

    RETURN:
    A tuple containing two objects: a list of optimal lambdas, one for each item,
    as well as the average Base Loop this choice of lambdas produced
    '''

    if len(some_simulation_result) == 0:
        return -1
    else:
        optimal_avg_baseloop = min(some_simulation_result.keys())
        optimal_lambda = some_simulation_result[optimal_avg_baseloop]

        return optimal_avg_baseloop, optimal_lambda


def get_optimal_path(some_simulation_result):
    if len(some_simulation_result) == 0:
        print('There is no answer')
    else:
        for i in sorted(some_simulation_result):
            print((i,some_simulation_result[i]))



def display_simulation_results(optimal_result):
    '''
    Displays the optimal lamdbas and average base loop found in the Simulation
    or indicate that no feasible solutions were found

    PARAMETERS:
    optimal_result := a tuple of the average base loop and its corresponding
                      lambdas

    RETURN:
    None
    '''
    print("***************************")
    print("Simulation Output:")
    if optimal_result != -1:
        print("Optimal Choice of Lambdas: {}".format(optimal_result[1]))
        print("Optimal average baseloop: {}".format(optimal_result[0]))
        print("***************************")

    else:
        print("No feasible solution found")
        print("***************************")



def discrete_descent(L, J, I0, h, a, trigger_points, D, t, Tau, T, num_simulation, optimal_lambda, neighbourhood,step,print_optimal_info):

    '''
    This function run a descent to find the optimal lambdas


    L: number of items
    J: number of time periods
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: A list of lists containing all item demands in each time period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    num_simulation: the number of simulations for each descent to find the local optimal in a neighborhood
    optimal_lambda: The optimal lambda from
    neighbourhood:
     step:
    :return:
    global_average_baseloop: return global optimal baseloop in this gradient
    global_lambda : return global optimal lambdas in this gradient
    '''


    #initiate the descent
    print('This is a discrete descent')
    init_average_baseloop=get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, optimal_lambda, t, Tau, T, False)
    print('The initial optimal lambdas are {} and the initial average base loop time is {}'.format(optimal_lambda,\
                                                                                                   init_average_baseloop))
    #Global optimal values in the descent
    global_average_baseloop=init_average_baseloop
    global_lambda = optimal_lambda

    #Check if it get stuck at a certain point
    copy_baseloop=init_average_baseloop
    repeat_count=0

    for i in range(step):
        # Check whether the new set of lambda changes
        if(print_optimal_info==True):
            print('Step: ',i+1)

        feasible_results = random_simulation(L, J, I0, h, a, trigger_points, D, t, Tau, T, num_simulation,
                                             optimal_lambda, neighbourhood)

        #Stop when the feasible dictionary is empty
        if(len(feasible_results)==0):
            break



        #Update optimal values
        optimal_avg_baseloop, optimal_lambda = get_optimal_siumulation_results(feasible_results)
        if (print_optimal_info == True):
            print('The average baseloop is {} and the optimal lambda is {}'.format( optimal_avg_baseloop,
                                                                                       optimal_lambda))

        # Check if it get stuck at a certain point
        if(copy_baseloop==optimal_avg_baseloop):
            repeat_count+=1
            if(repeat_count==10):
                break
        else:
            repeat_count=0
            copy_baseloop = optimal_avg_baseloop


        #update global optimal values
        if (global_average_baseloop>optimal_avg_baseloop):
            global_average_baseloop=optimal_avg_baseloop
            global_lambda=optimal_lambda

    print('**************')
    if (print_optimal_info == True):
        print('The optimal average base loop time is {} and the optimal lambdas are {}'.format(global_average_baseloop,\
                                                                                        global_lambda))
    #To print only
    global_average_baseloop = get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, global_lambda, t, Tau, T,
                                                      print_optimal_info)

    return global_average_baseloop,global_lambda




def discrete_descent_compare(L, J, I0, h, a, trigger_points, D, t, Tau, T, num_simulation, optimal_lambda, neighbourhood,step,print_optimal_info):


    #initiate the descent
    print('This is a discrete descent')
    init_average_baseloop,init_total_cost=get_average_baseloop_time_compare(L, J, I0, h, a, trigger_points, D, optimal_lambda, t, Tau, T, False)
    print('The initial optimal lambdas are {} and the initial average base loop time is {}'.format(optimal_lambda,\
                                                                                                   init_average_baseloop))
    #Global optimal values in the descent
    global_average_baseloop=init_average_baseloop
    global_lambda = optimal_lambda

    #Check if it get stuck at a certain point
    copy_baseloop=init_average_baseloop
    repeat_count=0
    compare=[[],[]] #average vs cost
    compare[0].append(init_average_baseloop)
    compare[1].append(init_total_cost)

    for i in range(step):
        # Check whether the new set of lambda changes
        if(print_optimal_info==True):
            print('Step: ',i+1)

        feasible_results = random_simulation(L, J, I0, h, a, trigger_points, D, t, Tau, T, num_simulation,
                                             optimal_lambda, neighbourhood)

        #Stop when the feasible dictionary is empty
        if(len(feasible_results)==0):
            break



        #Update optimal values
        optimal_avg_baseloop, optimal_lambda = get_optimal_siumulation_results(feasible_results)
        current_average_baseloop,current_total_cost=get_average_baseloop_time_compare(L, J, I0, h, a, trigger_points, D, optimal_lambda, t, Tau, T, False)
        compare[0].append(current_average_baseloop)
        compare[1].append(current_total_cost)
        if (print_optimal_info == True):
            print('The average baseloop is {} and the optimal lambda is {}'.format( optimal_avg_baseloop,
                                                                                       optimal_lambda))

        # Check if it get stuck at a certain point
        if(copy_baseloop==optimal_avg_baseloop):
            repeat_count+=1
            if(repeat_count==10):
                break
        else:
            repeat_count=0
            copy_baseloop = optimal_avg_baseloop


        #update global optimal values
        if (global_average_baseloop>optimal_avg_baseloop):
            global_average_baseloop=optimal_avg_baseloop
            global_lambda=optimal_lambda

    print('**************')
    if (print_optimal_info == True):
        print('The optimal average base loop time is {} and the optimal lambdas are {}'.format(global_average_baseloop,\
                                                                                        global_lambda))
    #To print only
    global_average_baseloop = get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, global_lambda, t, Tau, T,
                                                      print_optimal_info)


    plt.plot(compare[0],compare[1],marker='o')
    plt.show()
    plt.scatter(compare[0],compare[1],marker='o')
    plt.show()



def discrete_descent_cmc(L, J, I0, h, a, trigger_points, D, t, Tau, T, optimal_lambda, neighbourhood,step):

    '''
    This function run a descent to find the optimal lambdas


    L: number of items
    J: number of time periods
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: A list of lists containing all item demands in each time period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    num_simulation: the number of simulations for each descent to find the local optimal in a neighborhood
    optimal_lambda: The optimal lambda from
    neighbourhood:
     step:
    :return:
    global_average_baseloop: return global optimal baseloop in this gradient
    global_lambda : return global optimal lambdas in this gradient
    '''



    record=[]
    time_record=[]
    #initiate the descent
    print('This is a discrete descent')
    init_average_baseloop = get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, optimal_lambda, t, Tau, T, False)


    min_lambda=optimal_lambda
    min_baseloop=init_average_baseloop

    record.append(init_average_baseloop)
    current_lambda = optimal_lambda

    current_loop_time=init_average_baseloop
    print('Step 0')
    print('The average baseloop is {} and the optimal lambda is {}'.format(current_loop_time, current_lambda ))

    for i in range(step):

        print('step {}'.format(i+1))

        start = time.time()
        next_lambda = get_random_lambdas_mcmc_random(L, J, I0, h, a, trigger_points, D, current_lambda , t, Tau, T, False,neighbourhood)

        end = time.time()
        timetime=end-start
        print("time for search {}".format(timetime))

        next_loop_time = get_average_baseloop_time(L, J, I0, h, a, trigger_points, D, next_lambda, t, Tau, T, False)



        if next_loop_time<min_baseloop:
            min_lambda=next_lambda
            min_baseloop=next_loop_time


        time_record.append(timetime)

        current_loop_exp = np.exp(40 * current_loop_time)

        next_loop_exp = np.exp(next_loop_time*40)

        gamma = np.minimum(1,current_loop_exp/next_loop_exp)

        if(gamma == 1):

            print('The average baseloop is {} and the optimal lambda is {}'.format(next_loop_time, next_lambda))
            current_lambda = next_lambda
            current_loop_time = next_loop_time
        else:
            b=random.uniform(0, 1)
            if(b<gamma):
                print('The average baseloop is {} and the optimal lambda is {}'.format(next_loop_time, next_lambda))
                current_lambda = next_lambda
                current_loop_time = next_loop_time

            else:
                print('It is the same. The average baseloop is {} and the optimal lambda is {}'.format(current_loop_time, current_lambda))

        record.append(current_loop_time)




    print('this is min base loop {}'.format(min_baseloop))

    print('this is min base loop lambda {}'.format(min_lambda))


    print('this is the min avg {}'.format(min(record)))
    # plt.hist(record[300:], bins = 30)
    # plt.show()
    avg_timetime=sum(time_record)/len(time_record)
    print('this is avg time record {}'.format(avg_timetime))


def discrete_descent_cmc_global(L, J, I0, h, a, trigger_points, D, t, Tau, T, optimal_lambda, neighbourhood_sd,step):

    '''
    This function run a descent to find the optimal lambdas


    L: number of items
    J: number of time periods
    I0: a list of item initial inventories
    h: inventory cost
    a: changeover cost
    trigger_points: a list of item trigger points
    D: A list of lists containing all item demands in each time period
    t: a list of time takes to produce one unit of item
    Tau: cost tolerance
    T: the total time available to run the loop in each time period
    num_simulation: the number of simulations for each descent to find the local optimal in a neighborhood
    optimal_lambda: The optimal lambda from
    neighbourhood:
     step:
    :return:
    global_average_baseloop: return global optimal baseloop in this gradient
    global_lambda : return global optimal lambdas in this gradient
    '''

    record=[]

    surveyed_skipping={}
    surveyed_baseloop = {}

    #initiate the descent
    print('This is a discrete descent')
    min_baseloop_given, min_skipping_given = get_global(L, J, I0, h, a, trigger_points, D, optimal_lambda, t, Tau, T,
                                                        False)

    record.append(min_baseloop_given)
    current_lambda =  optimal_lambda

    current_loop_time=min_baseloop_given



    minmin_baseloop=min_baseloop_given
    minmin_lambda=optimal_lambda

    print('Step 0')
    print('The baseloop is {} and the global lambda is {}'.format(current_loop_time, current_lambda ))

    for i in range(step):

        print('step {}'.format(i+1))

        new_lambda=get_random_lambdas_mcmc_random(L, J, I0, h, a, trigger_points, D, current_lambda, t, Tau, T, False,neighbourhood_sd)
        new_min_baseloop_b, new_min_skipping_b = get_global(L, J, I0, h, a, trigger_points, D, new_lambda, t, Tau, T,
                                                            False)

        if minmin_baseloop>new_min_baseloop_b:
            minmin_baseloop = new_min_baseloop_b
            minmin_lambda =new_lambda


        current_loop_exp = np.exp(1000 * current_loop_time/J)

        next_loop_exp = np.exp(new_min_baseloop_b*1000/J)

        gamma = np.minimum(1,current_loop_exp/next_loop_exp)
        print('gamma is {}'.format(gamma))


        if(gamma == 1):
            print('baseloop is from {} to {}'.format(current_loop_time, current_loop_time))

            print('The baseloop is {} and the global lambda is {}'.format(new_min_baseloop_b, new_lambda))
            current_lambda = new_lambda
            current_loop_time = new_min_baseloop_b
        else:
            b=random.uniform(0, 1)
            if(b<gamma):
                print('baseloop is from {} to {}'.format(current_loop_time, new_min_baseloop_b))

                print('The baseloop is {} and the global lambda is {}'.format(new_min_baseloop_b, new_lambda))
                current_lambda =  new_lambda
                current_loop_time = new_min_baseloop_b
            else:
                print('It is the same. The current global baseloop is {} and the optimal lambda is {}'.format(current_loop_time, current_lambda))


        record.append(current_loop_time)
    print(min(record))

    print('min global baseloop is {} and min global lambda is {}'.format(minmin_baseloop, minmin_lambda))

   # plt.hist(record[300:], bins = 30)
   # plt.show()



def main():
    random.seed(0)

    csv_input = BaseLoopInputData('Input_Data3.csv')
    demand_schedule = csv_input.entire_demand_schedule
    unit_production_time = csv_input.all_production_times
    holding_cost = csv_input.inventory_cost
    num_items = len(holding_cost)
    num_periods = len(demand_schedule)
    demand_schedule_init = demand_schedule.copy()
    demand_schedule_init.insert(0, [0] * num_items)
    changeover_cost = csv_input.changeover_cost
    initial_inventory = csv_input.initial_inventories
    total_time = csv_input.total_time
    cost_tolerance = csv_input.cost_tolerance
    #trigger_points = csv_input.trigger_points
    trigger_points = [0] * num_items




    kwargs = {'num_items': num_items, 'num_periods': num_periods, \
              'unit_production_time': unit_production_time, \
              'total_time': total_time, 'initial_inventory': initial_inventory, \
              'demand_schedule': demand_schedule, 'cost_tolerance': cost_tolerance, \
              'changeover_cost': changeover_cost, 'holding_cost': holding_cost, \
              'demand_schedule_init': demand_schedule_init}

    optimal_lambdas = cost_model(**kwargs)
    if optimal_lambdas == -1:
        optimal_lambdas = [random.randint(1, 100) for i in range(num_items)]


    num_simulation = 100000
    neighbourhood = 10


    print('check lambda is {}'.format(optimal_lambdas))


    avg_baseloop = get_average_baseloop_time(num_items, num_periods, \
                                             initial_inventory, holding_cost, changeover_cost, trigger_points, \
                                             demand_schedule, optimal_lambdas, unit_production_time, cost_tolerance, \
                                             total_time, True)
    print('demand_schedule_init: ', demand_schedule_init)





    print('*********************')

        # Run original simulations from the paper
    feasible_results = random_simulation(num_items, num_periods, \
                                         initial_inventory, holding_cost,\
                                         changeover_cost, trigger_points, \
                                         demand_schedule, unit_production_time,\
                                         cost_tolerance, total_time, \
                                         num_simulation, optimal_lambdas, \
                                         neighbourhood)
    optimal_result = get_optimal_siumulation_results(feasible_results)
    display_simulation_results(optimal_result)
    print('*********************')


    # Print specific information of the current optimal answer from the original simulations
    print("this is the optimal answer")
    print("lambda: {}".format(optimal_result[1]))
    opt_baseloop = get_average_baseloop_time(num_items, num_periods, \
                                             initial_inventory, holding_cost, changeover_cost, trigger_points, \
                                             demand_schedule, optimal_result[1], unit_production_time, cost_tolerance, \
                                             total_time, True)





    print("this is from MCMC")
    '''
    opt_baseloop_mcmc = get_average_baseloop_time(num_items, num_periods, \
                                             initial_inventory, holding_cost, changeover_cost, trigger_points, \
                                             demand_schedule, [6, 36, 4, 2, 4, 5, 8, 6, 43], unit_production_time, cost_tolerance, \
                                             total_time, True)
    '''



    #Run the discrete descent
    '''
    num_iterations_in_descent = 500
    neighbourhood_range_in_descent=2
    num_samples_in_each_descent=200

    discrete_descent(num_items, num_periods, \
                                         initial_inventory, holding_cost,\
                                         changeover_cost, trigger_points, \
                                         demand_schedule, unit_production_time,\
                                         cost_tolerance, total_time, \
                                         num_iterations_in_descent, optimal_lambdas, \
                                         neighbourhood_range_in_descent,num_samples_in_each_descent,True)
                                         
    print('******************')                                         
    '''


    #Run MCMC Metropolis algorithm to find the global optimal


    '''
    
    start = time.time()
    print("hello")
    neighbourhood_in_descent_sd = 1
    num_iterations_in_descent = 300000

    discrete_descent_cmc(num_items, num_periods, \
                     initial_inventory, holding_cost, \
                     changeover_cost, trigger_points, \
                     demand_schedule, unit_production_time, \
                     cost_tolerance, total_time, \
                     optimal_result[1], \
                     neighbourhood_in_descent_sd, num_iterations_in_descent)
    end=time.time()
    print(end-start)
    '''
    print('*********************')
    get_average_baseloop_time(num_items, num_periods, \
                              initial_inventory, holding_cost, changeover_cost, trigger_points, \
                              demand_schedule, [2.300323086722109, 10.958711102899409, 1.565117103332754, 1.7692001885366753, 1.2605352580014204, 3.566499920843445], unit_production_time, cost_tolerance, \
                              total_time, True)

    print('*********************')



    '''
    print('************************************')

    get_global(num_items, num_periods, \
                     initial_inventory, holding_cost, \
                     changeover_cost, trigger_points, \
                     demand_schedule,[2.300323086722109, 10.958711102899409, 1.565117103332754, 1.7692001885366753, 1.2605352580014204, 3.566499920843445], unit_production_time, cost_tolerance ,total_time, True)

    get_average_baseloop_time(num_items, num_periods, \
                              initial_inventory, holding_cost, changeover_cost, trigger_points, \
                              demand_schedule,[2.300323086722109, 10.958711102899409, 1.565117103332754, 1.7692001885366753, 1.2605352580014204, 3.566499920843445], unit_production_time, cost_tolerance, \
                              total_time, True)

    '''



    '''

    get_average_baseloop_time(num_items, num_periods, \
                              initial_inventory, holding_cost, changeover_cost, trigger_points, \
                              demand_schedule, [6,36,4,2,4,5,8,6,43], unit_production_time, cost_tolerance, \
                              total_time, True)
    aa,ss,dd,ff,gg=explore_fully(2, num_items, initial_inventory, holding_cost, changeover_cost, \
                  trigger_points, demand_schedule, [6,36,4,2,4,5,8,6,43], unit_production_time, cost_tolerance, total_time, False)

    print('from explore fully {}'.format(aa))
    print('from explore fully {}'.format(ss))
    '''
    
    '''
                              
    print('******************')
    get_greedy_and_potential(num_items, num_periods, \
                              initial_inventory, holding_cost, changeover_cost, trigger_points, \
                              demand_schedule, [3,14,2,5,2,2], unit_production_time, cost_tolerance, \
                              total_time, True)
    print('******************')
    
    print(demand_schedule[2])
    '''



    print('******************')

    '''
    start=time.time()
    get_global(num_items, num_periods, \
                     initial_inventory, holding_cost, \
                     changeover_cost, trigger_points, \
                     demand_schedule, [3,14,2,5,2,2], unit_production_time, cost_tolerance ,total_time, True)

    end = time.time()
    timetime=end-start
    print("time used is {}".format(timetime))
    '''



    #'''
    print('*********************')
    print('*********************')
    print('*********************')
    print('*********************')
    
    S=[[1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 1, 1],
       [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]


    get_cost(S, num_items, num_periods, \
                     initial_inventory, holding_cost, \
                     changeover_cost, trigger_points, \
                     demand_schedule, [2.3065814835177068, 11.176917017048623, 1.5799521633529656, 1.8394751475130202, 1.2618590561267395, 3.6001293522290347], unit_production_time, cost_tolerance ,total_time, True)

    #'''


    print('*********************')
    '''
    start=time.time()
    get_neighbor_global(num_items, num_periods, \
                     initial_inventory, holding_cost, \
                     changeover_cost, trigger_points, \
                     demand_schedule, [6,36,4,2,4,5,8,6,43], unit_production_time, cost_tolerance ,total_time, False,1,20)

    end = time.time()
    timetime= end-start
    print('time is {}'.format(timetime))
    '''
    print('******************')


    '''
    start = time.time()
    neighbourhood_in_descent_sd = 1
    num_iterations_in_descent = 2000

    discrete_descent_cmc_global(num_items, num_periods, \
                         initial_inventory, holding_cost, \
                         changeover_cost, trigger_points, \
                         demand_schedule, unit_production_time, \
                         cost_tolerance, total_time, \
                         [3.823033596023957, 16.491443798099127, 2.29288339456614, 2.4119982966205042, 1.907220126997165, 2.639994452667621], \
                         neighbourhood_in_descent_sd, num_iterations_in_descent)



    end = time.time()
    timetime= end-start
    print('time is {}'.format(timetime))
    
    
    '''
    '''

    start=time.time()
    get_global(num_items, num_periods, \
                     initial_inventory, holding_cost, \
                     changeover_cost, trigger_points, \
                     demand_schedule, [3, 19, 2, 1, 2, 3, 4, 3, 22], unit_production_time, cost_tolerance ,total_time, False)

    end = time.time()
    timetime=end-start
    print("time used is {}".format(timetime))

    get_greedy_and_potential(num_items, num_periods, \
                     initial_inventory, holding_cost, \
                     changeover_cost, trigger_points, \
                     demand_schedule, [3, 19, 2, 1, 2, 3, 4, 3, 22], unit_production_time, cost_tolerance ,total_time, True)
    '''
    # big test
    '''
    unit_production_time_np=np.array(unit_production_time).reshape((-1,1))
    Lambda_np=np.array([6, 36, 4, 2, 4, 5, 8, 6, 43])
    Lambda_np=np.reshape(Lambda_np,(-1,1))
    skipping_np=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 0, 1, 1, 0],\
                          [1, 0, 1, 1, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0], \
                          [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1, 1, 0, 0], \
                          [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    skipping_np=np.transpose(skipping_np)
    print("this is unit{}".format(unit_production_time_np))
    print( unit_production_time_np.shape)


    bound = sigma(skipping_np, unit_production_time_np, Lambda_np)
    print("bound is {}".format(bound))
    demand_schedule_np = np.transpose(np.array(demand_schedule))


    print("check check demand {}".format(demand_schedule_np))
    initial_inventory_np = np.reshape(np.array(initial_inventory),(-1,1))
    print("{} {} {}".format(bound.shape,demand_schedule_np.shape,initial_inventory_np.shape))
    print(demand_schedule_np)
    print(greedy(demand_schedule_np[:,0],np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])))
    print(produce( greedy(demand_schedule_np[:,0],np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])), unit_production_time_np,Lambda_np, total_time))

    all_candidate, all_inventory=explore( 2, unit_production_time_np, Lambda_np, total_time, demand_schedule_np, initial_inventory_np)
    print("all")
    for i in range(len(all_candidate)):
        print("All Candidate {} is {}".format(i,all_candidate[i]))
        print("All current base loop for {} is {}".format(i,sigma(all_candidate[i],unit_production_time_np, Lambda_np)))
        print( "all_inventory for {} is {}".format(i, all_inventory[i]))
    print("fAll Candidate {} is {}".format(0, all_candidate[0]))


    under_bound_candidate, under_bound_inventory=explore_and_bound(bound, 4, unit_production_time_np, Lambda_np, total_time, demand_schedule_np, initial_inventory_np)
    print(len(under_bound_candidate))
    print("underbound")
    for i in range(len(under_bound_candidate)):
        print("Candidate {} is {}".format(i,under_bound_candidate[i]))
        print("current base loop for {} is {}".format(i,sigma(under_bound_candidate[i],unit_production_time_np, Lambda_np)))
        print("current_inventory for {} is {}".format(i, under_bound_inventory[i]))
    '''

    #small test

    '''
    unit_production_time_np=np.array(unit_production_time).reshape((-1,1))
    Lambda_np=np.array([4, 19, 4, 1, 5, 3])
    Lambda_np=np.reshape(Lambda_np,(-1,1))
    skipping_np=np.array([[1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0],\
     [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0],\
      [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
    skipping_np=np.transpose(skipping_np)
    print("this is unit{}".format(unit_production_time_np))
    print( unit_production_time_np.shape)


    bound = sigma(skipping_np, unit_production_time_np, Lambda_np)
    print("bound is {}".format(bound))
    demand_schedule_np = np.transpose(np.array(demand_schedule))


    print("check check demand {}".format(demand_schedule_np))
    initial_inventory_np = np.reshape(np.array(initial_inventory),(-1,1))
    print("{} {} {}".format(bound.shape,demand_schedule_np.shape,initial_inventory_np.shape))
    print(demand_schedule_np)
    print(greedy(demand_schedule_np[:,0],np.array([0, 0, 0, 0, 0, 0])))
    print(produce( greedy(demand_schedule_np[:,0],np.array([0, 0, 0, 0, 0, 0])), unit_production_time_np,Lambda_np, total_time))

    all_candidate, all_inventory=explore( 2, unit_production_time_np, Lambda_np, total_time, demand_schedule_np, initial_inventory_np)
    print("all")
    for i in range(len(all_candidate)):
        print("All Candidate {} is {}".format(i,all_candidate[i]))
        print("All current base loop for {} is {}".format(i,sigma(all_candidate[i],unit_production_time_np, Lambda_np)))
        print( "all_inventory for {} is {}".format(i, all_inventory[i]))
    print("fAll Candidate {} is {}".format(0, all_candidate[0]))

    print("bound is {}".format(bound))
    print("underbound")
    under_bound_candidate, under_bound_inventory=explore_and_bound(bound, 2, unit_production_time_np, Lambda_np, total_time, demand_schedule_np, initial_inventory_np)
    print(len(under_bound_candidate))

    for i in range(len(under_bound_candidate)):
        print("Candidate {} is {}".format(i,under_bound_candidate[i]))
        print("current base loop for {} is {}".format(i,sigma(under_bound_candidate[i],unit_production_time_np, Lambda_np)))
        print("current_inventory for {} is {}".format(i, under_bound_inventory[i]))

    print('************************')
    '''




    print('test area')
    '''
    Lambda_np_init=np.array(optimal_result[1]).reshape((-1,1))
    num_items_np, num_periods_np, unit_production_time_np, total_time_np, \
    initial_inventory_np, demand_schedule_np, cost_tolerance_np, changeover_cost_np, \
    holding_cost_np, demand_schedule_init_np=np_transform(**kwargs)

    if_greedy_ok(unit_production_time_np, Lambda_np_init, total_time_np, demand_schedule_np, initial_inventory_np, changeover_cost_np, holding_cost_np, cost_tolerance_np)


    print(greedy_left(unit_production_time_np, Lambda_np_init, total_time_np, demand_schedule_np, initial_inventory_np))
    print("check MCMC")
    mcmc_np(initial_inventory_np, holding_cost_np, changeover_cost_np, demand_schedule_np, Lambda_np_init, unit_production_time_np, cost_tolerance, total_time_np,\
            1, 400)
    '''

    #baseloop_added, skipping_added=greedy_left(unit_production_time_np, np.transpose([[4, 19, 4, 1, 5, 3]]), total_time_np, demand_schedule_np, initial_inventory_np)


    #print(baseloop_added/12, skipping_added)
    #a=np.transpose([[1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],\
                    #[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
    #print(sigma(a,unit_production_time_np,np.transpose([[5, 24, 2, 4, 2, 3]])))

    '''
    start = time.time()
    get_average_baseloop_time_compare(6, 12, initial_inventory, holding_cost, changeover_cost, trigger_points, demand_schedule, [4, 19, 4, 1, 5, 3], unit_production_time, cost_tolerance, total_time, True)


    end=time.time()
    timetime=end-start
    print('this is reg {}'.format(timetime))

    

    start=time.time()
    if_greedy_ok(unit_production_time_np,np.transpose([[4, 19, 4, 1, 5, 3]]), total_time_np, demand_schedule_np, initial_inventory_np, changeover_cost_np, holding_cost_np, cost_tolerance)

    end=time.time()
    timetime=end-start
    print('this is np {}'.format(timetime))
    '''
    # find global
    '''
    start = time.time()
    print("hello")
    global_baseloop, global_skipping = get_global(unit_production_time_np, np.transpose([[4, 19, 4, 1, 5, 3]]), total_time_np, demand_schedule_np,
                                                  initial_inventory_np)
    print(global_baseloop, global_skipping)
    print(if_this_skipping_ok(global_skipping,unit_production_time_np,  np.transpose([[4, 19, 4, 1, 5, 3]]), total_time_np,\
                              demand_schedule_np, initial_inventory_np,changeover_cost_np,holding_cost_np,cost_tolerance))

    end = time.time()
    print(end - start)
    '''



if __name__ == "__main__":



    main()
