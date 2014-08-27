import random
random.seed()

def true_TD_fn(domain, settings):

    num_episodes   = settings['num_episodes']
    num_runs       = settings['num_runs']
    alpha         = settings['alpha']
    lambda0       = settings['lambda']

    num_features   = domain['num_features']
    num_states     = domain['num_states']
    gamma         = domain['gamma']
    V_true        = domain['V_true']
    p_right       = domain['p_right']
    phi_table     = domain['phi_table']
    reward_table  = domain['reward_table']
    

    RMS_per_run = [0 for i in range(num_runs)]
    for run in range(num_runs):
    
        theta = [0 for i in range(num_features)]
        RMS_per_episode = 0
        for episode in range(num_episodes):
        
            s = num_states-1
        
            e = [0 for i in range(num_features)]
            Vcurrent = 0
            for i in range(num_features):
                Vcurrent += theta[i]*phi_table[i][s]
        
            while(s != 0):
            #########################################################
                # generate next state and reward
                if (random.random() < p_right):
                    s2 = s-1
                    r = reward_table[s][1]
                else:
                    if (s == num_states-1):
                        s2 = s
                    else:
                        s2 = s+1
                    r = reward_table[s][0]
                    
                Vnext = 0
                for i in range(num_features):
                    Vnext += theta[i]*phi_table[i][s2] # thisQ
            
                delta = r + gamma*Vnext - Vcurrent
            
                Vs = 0
                for i in range(num_features):
                    Vs += theta[i]*phi_table[i][s]
                delta2 = Vcurrent - Vs
            
                ephi = 0
                for i in range(num_features):
                    ephi += e[i]*phi_table[i][s]
                
                for i in range(num_features):
                    e[i] = gamma*lambda0*e[i] + alpha*phi_table[i][s]\
                     - alpha*gamma*lambda0*ephi*phi_table[i][s]
                    
                    theta[i] += delta*e[i] + alpha*delta2*phi_table[i][s]
            
            
                Vcurrent = Vnext
                s = s2
                ##################################################################
        

            SE = 0
            for ss in range(1,num_states):
                Vss = 0
                for i in range(num_features):
                    Vss += theta[i]*phi_table[i][ss]
                SE += (Vss - V_true[ss])**2
            RMS = (SE/float(num_states-1))**0.5        
        
            beta = 1/float(episode+1)
            RMS_per_episode = (1-beta)*RMS_per_episode + beta*RMS
    
     
        RMS_per_run[run] = RMS_per_episode


    avg_RMS = 0
    for i in range(num_runs):
        avg_RMS += RMS_per_run[i]
    avg_RMS /= float(num_runs)


    error = 0;
    for run in range(num_runs):
        beta = 1/float(run+1)
        error = (1-beta)*error + beta*((RMS_per_run[run] - avg_RMS)**2)
    error = (error/float(num_runs))**0.5

    return (avg_RMS, error)