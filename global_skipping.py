import numpy as np
from find_skipping_coeff import *
import time
#implement numpy

def produce( skipping_coeff_at_this_period, t, Lambda, T):
    if(all(skipping_coeff_at_this_period==0)):
        return 0
    else:
        Lambda_base=np.multiply(Lambda,skipping_coeff_at_this_period.reshape(-1,1))
        baseloop=np.dot(t.flatten(),Lambda_base.flatten())
        num_round=np.floor(T/baseloop)
        product=num_round*(np.multiply(skipping_coeff_at_this_period,Lambda))

       # print("baseloop {}".format(baseloop))
       # print("Lambda {}".format(Lambda))
       # print("skipping_coeff_at_this_period {}".format(skipping_coeff_at_this_period))

       # print("t is {} and the production is {}".format(T,product))

    return product

def greedy(demand_in_this_period,current_inventory):

    skipping_at_least = demand_in_this_period>current_inventory
    return skipping_at_least


def sigma(skipping_given,t,Lambda):

    baseloop_time= np.dot(skipping_given.sum(axis=1),np.multiply(t,Lambda))
    return baseloop_time

def remainder(demand_in_this_period,all_demand_after_this,current_inventory,t, Lambda, T):

    current_skipping_coeff=greedy(demand_in_this_period, current_inventory)

    production=produce( current_skipping_coeff, t, Lambda, T)

    next_inventory=current_inventory+production-demand_in_this_period

    next_skipping_coeff=greedy(all_demand_after_this, next_inventory)

    return next_skipping_coeff

def potential_optimum(skipping_given,t, Lambda, T,demand,current_inventory):

    num_period_coeff_given= skipping_given.shape[1]
    skipping_coeff_current =greedy(demand[:,[num_period_coeff_given]],current_inventory)

    all_demand_after_this=np.sum(demand[:,num_period_coeff_given:demand.shape[1]],axis=1).reshape(-1,1)
    next_skipping_coeff = remainder(demand[:,[num_period_coeff_given]],all_demand_after_this,current_inventory,t, Lambda, T)
   # print("demand check {}".format(demand[:,num_period_coeff_given:demand.shape[1]]))
    #print("num_period_coeff_given is {}".format(num_period_coeff_given))
    #print("skipping_coeff_current is {}".format(skipping_coeff_current))
   # print("all_demand_after_this is {}".format(all_demand_after_this))
   # print("next_skipping_coeff is {}".format(next_skipping_coeff))
    potential_baseloop=sigma(skipping_given,t,Lambda)+\
                       sigma(skipping_coeff_current,t,Lambda)+\
                       sigma(next_skipping_coeff,t,Lambda)

    return potential_baseloop


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def skipping_cadidate(t, Lambda, T,demand_in_this,initial_inventory):

    L=demand_in_this.shape[0]
   # print("demand in this is {}".format(demand_in_this))
    #print("initial_inventory {}".format(initial_inventory))

    skipping_required=greedy(demand_in_this,initial_inventory)

    #print("skipping_required is {}".format(skipping_required))
    candidate=[]
    new_inventory=[]
   # print("this {} is {}".format(skipping_required,np.sum(skipping_required)))
    num_cadi=2**(L-np.sum(skipping_required))
    index_choice=np.where(skipping_required==0)[0]
    #print("This is num cadi")


    for i in range( num_cadi):
        alt_skipping_coeff=np.copy(skipping_required)
        index=bin_array(i,L-np.sum(skipping_required))

        alt_skipping_coeff[index_choice]=index.reshape(-1,1)

        next_inventory=initial_inventory+produce( alt_skipping_coeff, t, Lambda, T)-demand_in_this

        #print("initial_inventory {}".format(initial_inventory))
       # print("produce( alt_skipping_coeff, t, Lambda, T) {}".format(produce( alt_skipping_coeff, t, Lambda, T)))
       # print("demand_in_this {}".format(demand_in_this))
        alt_skipping_coeff=np.reshape(alt_skipping_coeff,(-1,1))

        next_inventory=np.reshape(next_inventory,(-1,1))


        #print("alt_skipping_coeff {}".format(alt_skipping_coeff))
        if all(next_inventory>=0):
            candidate.append(alt_skipping_coeff)
            new_inventory.append(next_inventory)


    return candidate, new_inventory

def explore(level,t, Lambda, T,demand,initial_inventory):

    if level==1:
        #print("this is demand {}".format(demand[:, [0]]))
        candidate, new_inventory = skipping_cadidate(t, Lambda, T, demand[:, [0]], initial_inventory)

        #print("this is new inventory {}".format(new_inventory))
        return candidate, new_inventory
    else:
        previous_candidate, previous_inventory=explore(level-1,t, Lambda, T,demand,initial_inventory)
        n=len(previous_candidate)
        all_candidate_in_this =[]
        all_inventory_in_this = []

        for i in range(n):

            current_candidate,current_inventory=skipping_cadidate(t, Lambda, T, demand[:, [level-1] ], previous_inventory[i])
            m=len(current_candidate)
            for j in range(m):
                added_candi=np.concatenate((previous_candidate[i],current_candidate[j]),axis=1)
                all_candidate_in_this.append(added_candi)
                all_inventory_in_this.append(current_inventory[j])

        return all_candidate_in_this,all_inventory_in_this



def explore_and_bound(bound,level,t, Lambda, T,demand,initial_inventory):


    all_candidate_in_this, all_inventory_in_this=explore(level,t, Lambda, T,demand,initial_inventory)

    n=len(all_inventory_in_this)
    under_bound_candidate=[]
    under_bound_inventory=[]
    #print("All candidate {}".format(all_candidate_in_this))
    #print("All candidate 0{}".format(all_candidate_in_this[0]))
    #print("All inventory 0{}".format(all_inventory_in_this[0]))

    for i in range(n):

        #print("THis is potential {}  {}".format(i,potential_optimum(all_candidate_in_this[i], t, Lambda, T, demand, all_inventory_in_this[i])))
        if potential_optimum(all_candidate_in_this[i], t, Lambda, T, demand, all_inventory_in_this[i])<bound:
            under_bound_candidate.append(all_candidate_in_this[i])
            under_bound_inventory.append(all_inventory_in_this[i])

    print("size of candidate is{} ".format(len(under_bound_candidate)))
    return under_bound_candidate, under_bound_inventory

def explore_next_and_bound(skipping_candidate,inventory_candidate,bound,t, Lambda, T,demand):

    # level=j
    level=skipping_candidate[0].shape[1]
    N=len(skipping_candidate)
    skipping_cadidate_added=[]
    inventory_cadidate_added = []
    candidate_greedy_baseloop =[]
    candidate_upper_b_baseloop=[]
    min_check=bound
    min_skipping=None


    for i in range(N):
        current_candidate, current_inventory = skipping_cadidate(t, Lambda, T, demand[:, [level]], inventory_candidate[i])


        m = len(current_candidate)
        for j in range(m):

            added_candi = np.concatenate((skipping_candidate[i], current_candidate[j]), axis=1)

            baseloop_added, skipping_added=greedy_left(t, Lambda, T,demand[:,level+1:12],current_inventory[j])

            check_skipping_full_with_g=np.concatenate((added_candi, skipping_added), axis=1)

            upper_b_baseloop=potential_optimum(added_candi, t, Lambda, T, demand, current_inventory[j])

            greedy_full_baseloop=sigma(added_candi,t,Lambda)+baseloop_added

            if greedy_full_baseloop<min_check:
                min_check=greedy_full_baseloop
                min_skipping=np.concatenate((added_candi, skipping_added), axis=1)

                print('this is min skipping {}'.format(min_skipping))

            if upper_b_baseloop < bound:
                skipping_cadidate_added.append(added_candi)
                inventory_cadidate_added.append(current_inventory[j])
                candidate_greedy_baseloop.append(greedy_full_baseloop)
                candidate_upper_b_baseloop.append(upper_b_baseloop)

    reduced_skipping_cadidate=[]
    reduced_inventory_candidate=[]
    K=len(skipping_cadidate_added)

    print('this is upper {}'.format(upper_b_baseloop))

    for i in range(K):
        if candidate_upper_b_baseloop[i]<min_check:
            reduced_skipping_cadidate.append(skipping_cadidate_added[i])
            reduced_inventory_candidate.append(inventory_cadidate_added[i])



    return reduced_skipping_cadidate, reduced_inventory_candidate, min_check,min_skipping


def explore_next_and_bound_and_check(skipping_candidate, inventory_candidate, bound, t, Lambda, T, demand):
    # level=j
    level = skipping_candidate[0].shape[1]
    N = len(skipping_candidate)
    skipping_cadidate_added = []
    inventory_cadidate_added = []
    candidate_greedy_baseloop = []
    candidate_upper_b_baseloop = []
    min_check = bound
    min_skipping = None

    for i in range(N):
        current_candidate, current_inventory = skipping_cadidate(t, Lambda, T, demand[:, [level]],
                                                                 inventory_candidate[i])

        m = len(current_candidate)
        for j in range(m):

            added_candi = np.concatenate((skipping_candidate[i], current_candidate[j]), axis=1)


            baseloop_added, skipping_added = greedy_left(t, Lambda, T, demand[:, level + 1:12], current_inventory[j])

            check_skipping_full_with_g = np.concatenate((added_candi, skipping_added), axis=1)

            upper_b_baseloop = potential_optimum(added_candi, t, Lambda, T, demand, current_inventory[j])

            greedy_full_baseloop = sigma(added_candi, t, Lambda) + baseloop_added

            if greedy_full_baseloop < min_check:
                min_check = greedy_full_baseloop
                min_skipping = np.concatenate((added_candi, skipping_added), axis=1)

                print('this is min skipping {}'.format(min_skipping))

            if upper_b_baseloop < bound:
                skipping_cadidate_added.append(added_candi)
                inventory_cadidate_added.append(current_inventory[j])
                candidate_greedy_baseloop.append(greedy_full_baseloop)
                candidate_upper_b_baseloop.append(upper_b_baseloop)

    reduced_skipping_cadidate = []
    reduced_inventory_candidate = []
    K = len(skipping_cadidate_added)

    print('this is upper {}'.format(upper_b_baseloop))

    for i in range(K):
        if candidate_upper_b_baseloop[i] < min_check:
            reduced_skipping_cadidate.append(skipping_cadidate_added[i])
            reduced_inventory_candidate.append(inventory_cadidate_added[i])

    return reduced_skipping_cadidate, reduced_inventory_candidate, min_check, min_skipping


def greedy_left(t, Lambda, T,demand_needed,initial_inventory):

    L=demand_needed.shape[0]
    J_left=demand_needed.shape[1]

    baseloop_added=0
    skipping_added=np.zeros([L,J_left])


    for i in range(J_left):

        skipping_required=demand_needed[:,[i]]>initial_inventory
        next_inventory = initial_inventory + produce(skipping_required, t, Lambda, T) - demand_needed[:,[i]]



        skipping_added[:,[i]]=skipping_required
        baseloop_added = baseloop_added+sigma(skipping_required,t,Lambda)

        initial_inventory=next_inventory


    return baseloop_added, skipping_added

def greedy_left_no_skipping(t, Lambda, T,demand_needed,initial_inventory):

    L=demand_needed.shape[0]
    J_left=demand_needed.shape[1]

    baseloop_added=0



    for i in range(J_left):
        skipping_required=demand_needed[:,[i]]>initial_inventory
        next_inventory = initial_inventory + produce(skipping_required, t, Lambda, T) - demand_needed[:,[i]]



        baseloop_added = baseloop_added+sigma(skipping_required,t,Lambda)

        initial_inventory=next_inventory


    return baseloop_added



def get_global(t, Lambda, T,demand,initial_inventory):

    level=2
    baseloop_full_g,skipping_full_g =greedy_left(t, Lambda, T,demand,initial_inventory)

    all_candidate_in_this, all_inventory_in_this = explore_and_bound(baseloop_full_g,level,t, Lambda, T,demand,initial_inventory)

    bound_f_greedy=sigma(skipping_full_g,t,Lambda)

    print("this is from greedy {} and {}".format(baseloop_full_g,skipping_full_g))


    for i in range(level,10):
        all_candidate_in_this, all_inventory_in_this, min_check, min_skipping=explore_next_and_bound(all_candidate_in_this,\
                                                                                                               all_inventory_in_this,bound_f_greedy,t, Lambda, T,demand)

        print('this is step {}'.format(i))
        if len(all_candidate_in_this)==0:
            print('The global optimum is found')
            print('The baseloop is {}. And the skipping coef is {}'.format(bound_f_greedy,skipping_full_g))
            return bound_f_greedy, skipping_full_g

        else:
            bound_f_greedy=min_check
            if not (min_skipping is None):
                skipping_full_g=min_skipping


def np_transform(num_items, num_periods, unit_production_time, total_time, \
               initial_inventory, demand_schedule, cost_tolerance, \
               changeover_cost, holding_cost, demand_schedule_init):
    num_items_np=num_items
    num_periods_np=num_periods
    unit_production_time_np=np.array(unit_production_time).reshape((-1, 1))
    total_time_np=total_time
    initial_inventory_np=np.array(initial_inventory).reshape((-1, 1))
    demand_schedule_np = np.transpose(np.array(demand_schedule))
    cost_tolerance_np=cost_tolerance
    changeover_cost_np=np.array(changeover_cost).reshape((-1,1))
    holding_cost_np=np.array(holding_cost).reshape((-1,1))
    demand_schedule_init_np=np.transpose(np.array(demand_schedule_init))





    return num_items_np, num_periods_np, unit_production_time_np, total_time_np, \
               initial_inventory_np, demand_schedule_np, cost_tolerance_np, \
               changeover_cost_np, holding_cost_np, demand_schedule_init_np


def if_this_skipping_ok(skipping_coeff,t, Lambda, T, demand_needed, initial_inventory,changeover_cost_np,holding_cost_np,cost_tolerance):
    total_changeover_cost = 0
    total_inventory_cost = 0
    J = demand_needed.shape[1]

    for i in range(J):

        if sum(skipping_coeff[:, i]) > 1:
            Lambda_base = np.multiply(Lambda, skipping_coeff[:, [i]])
            baseloop = np.dot(t.flatten(), Lambda_base.flatten())
            num_round = np.floor(T / baseloop)

            total_changeover_cost = total_changeover_cost + num_round * np.dot(changeover_cost_np.flatten(),
                                                                               skipping_coeff[:, [i]].flatten())

        leftover = initial_inventory + produce(skipping_coeff[:, [i]], t, Lambda, T) - demand_needed[:, [i]]

        total_inventory_cost = total_inventory_cost + np.dot(holding_cost_np.flatten(), leftover)
        initial_inventory = np.copy(leftover)

        if any(initial_inventory.flatten() < 0):
            #print('it is not okay here 1')
            return False


    total_cost = total_inventory_cost + total_changeover_cost

    if total_cost <= cost_tolerance:
        return True
    else:
        #print('this is total_changeover cost {}'.format(total_changeover_cost))
        #print('this is total_inventory cost {}'.format(total_inventory_cost))
        #print('it is not okay here 2')
        #print('total cost {}'.format(total_cost))
        #print('cost_tolerance: {}'.format(cost_tolerance))
        return False



def if_greedy_ok(t, Lambda, T, demand_needed, initial_inventory,changeover_cost_np,holding_cost_np,cost_tolerance):

    total_changeover_cost=0
    total_inventory_cost=0
    J=demand_needed.shape[1]

    for i in range(J):

        skipping_this_period=demand_needed[:,[i]]>initial_inventory


        if sum(skipping_this_period)>1:

            Lambda_base=np.multiply(Lambda,skipping_this_period)

            baseloop=np.dot(t.flatten(),Lambda_base.flatten())

            num_round=np.floor(T/baseloop)

            total_changeover_cost=total_changeover_cost+num_round*np.dot(changeover_cost_np.flatten(),skipping_this_period.flatten())



        leftover = initial_inventory+produce( skipping_this_period, t, Lambda, T)-demand_needed[:,[i]]

        total_inventory_cost=total_inventory_cost+np.dot(holding_cost_np.flatten(),leftover)
        initial_inventory=leftover

        if any(initial_inventory.flatten()<0):
            #print('it is not okay here 1')
            #print(i)
            #print(initial_inventory)
            return False

    #print('this is total_changeover cost {}'.format(total_changeover_cost))
    #print('this is total_inventory cost {}'.format(total_inventory_cost))
    total_cost=total_inventory_cost+total_changeover_cost
    #print('This is total cost {}'.format(total_cost))
    #print('cost_tolerance: {}'.format(cost_tolerance))
    if total_cost<=cost_tolerance:
        return True
    else:
        #print('it is not okay here 2')
        return False


def get_random_lambdas_normal_np(optimal_lambda, neighborhood_sd):

    new_lambda = optimal_lambda.copy()

    for i in range(optimal_lambda.shape[0]):

        generated_val = -1
        while generated_val <= 0:
            generated_val = int(np.random.normal(optimal_lambda[i,0],neighborhood_sd))
        new_lambda[i,0] = generated_val

    #print('this is new lambda {}'.format(new_lambda))
    return new_lambda

def get_random_lambdas_mcmc_np( initial_inventory, holding_cost_np, changeover_cost_np, Demand, optimal_lambda, t, cost_tolerance, T,neighborhood_sd):

    new_lambda = get_random_lambdas_normal_np(optimal_lambda, neighborhood_sd)

    while( not if_greedy_ok(t, new_lambda, T, Demand, initial_inventory,changeover_cost_np,holding_cost_np,cost_tolerance)):

        new_lambda = get_random_lambdas_normal_np(optimal_lambda, neighborhood_sd)

    return new_lambda



def mcmc_np(initial_inventory, holding_cost_np, changeover_cost_np, Demand, optimal_lambda, t, cost_tolerance, T,neighborhood_sd,step):


    record=[]
    time_record=[]
    #initiate the descent
    print('This is MCMC')
    baseloop_added, skipping_added=greedy_left(t, optimal_lambda, T, Demand, initial_inventory)

    init_baseloop=baseloop_added

    record.append(init_baseloop)
    current_lambda = optimal_lambda

    current_loop_time=init_baseloop



    #print('Step 0')
    #print('The average baseloop is {} and the optimal lambda is {}'.format(current_loop_time, current_lambda ))

    for i in range(step):

        print('step {}'.format(i+1))
        start = time.time()

        next_lambda = get_random_lambdas_mcmc_np( initial_inventory, holding_cost_np, changeover_cost_np,\
                                                  Demand, current_lambda, t, cost_tolerance, T,neighborhood_sd)

        end = time.time()
        timetime = end - start
        time_record.append(timetime)
        #print('time for search {}'.format(timetime))



        next_loop_time = greedy_left_no_skipping(t, next_lambda, T, Demand, initial_inventory)

        current_loop_exp = np.exp(5/12 * current_loop_time)

        next_loop_exp = np.exp(next_loop_time*5/12)

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

    print(min(record))
    global_min_greedy=min(record)
    print(isinstance(global_min_greedy,float))
    avg_time= sum(time_record)/len(time_record)
    print('this is avg time record {}'.format(avg_time))


    return global_min_greedy


def get_global_neighbor(L, J, I0, h, a, trigger_points, D, t, Tau, T, num_simulation, optimal_lambda, neighbourhood):
    feasible_results=random_simulation(L, J, I0, h, a, trigger_points, D, t, Tau, T, num_simulation, optimal_lambda, neighbourhood)


def main():
    a=np.array([[100],[2],[3]])
    print(get_random_lambdas_normal_np(a,1))


if __name__ == "__main__":
    main()