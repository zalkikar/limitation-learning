#code for training 
import torch
import numpy as np
from utils.utils import get_entropy, log_prob_density

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def get_entropy(mu, std):
    dist = Normal(mu, std)
    entropy = dist.entropy().mean()
    return entropy

def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                     - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)

def get_reward(discrim, state, action):
    """
    The reward function according to irl. It's log D(s,a). 
    
    Reward is higher the closer this is to 0, because the more similar it is to an expert action. :
    Is quite close to imitation learning, but hope here is that with such a large number of expert demonstrations and entropy bonuses etc. it learns more than direct imitation. 
    """
    state = torch.Tensor(state)
    try:
        action = torch.Tensor(action) # turn state into a tensor if not already
    except:
        pass
    state_action = torch.cat([state, action]) #TODO
    with torch.no_grad():
        return -math.log(discrim(state_action)[0].item())

def save_checkpoint(state, filename):


def train_discrim(discrim, memory, discrim_optim, demonstrations, discrim_update_num, batch_size, clip_param):
    """
    Training the discriminator. 

    Use binary cross entropy to classify whether or not a sequence was predicted by the expert (real data) or actor. 
    """
    memory = np.array(memory)  # s a r s' tuple
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) #actions taken by actor/policy
    expert_actions = list(memory[:, 4]) #actions taken by the expert # TODO

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
        
    criterion = torch.nn.BCELoss() # classify
 
    for _ in range(discrim_update_num):
        
        learner = discrim(torch.cat([states, actions], dim=1)) #pass (s,a) through discriminator
        
        # TODO
        demonstrations = torch.Tensor([states, expert_actions]) # pass (s,a) of expert through discriminator
        index = torch.randperm(demonstrations.shape[0])
        demonstrations_batch = demonstrations[index,:][:int(batch_size)] # batch of expert samples. Initially this was for the entire dataset, which might be too big. 
        expert = discrim(demonstrations_batch) #discrimator "guesses" whether or not these 
        # actions came from expert or learner
        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                        criterion(expert, torch.zeros((demonstrations_batch.shape[0], 1)))
                # discrim loss: predict agent is all wrong, get as close to 0, and predict expert is 1, getting as close to 1 as possible. 
        discrim_optim.zero_grad() # gan loss, it tries to always get it right. 
        discrim_loss.backward()
        discrim_optim.step()

        # take these steps, do it however many times specified. 

    expert_acc = ((discrim(demonstrations_batch) < 0.5).float()).mean() #how often it realized the fake examples were fake
    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.5).float()).mean() #how often if predicted expert correctly. 

    return expert_acc, learner_acc # accuracy, it's the same kind, but because imbalanced better to look at separately. 
 

def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, actor_critic_update_num, batch_size, clip_param):
    """
    Take a PPO step or two to improve the actor critic model,  using GAE to estimate returns. 
    
    In our case each trajectory it most one step, so the value function will have to do. 
    
    
    """
    memory = np.array(memory) 
    # tuple of a regular old RL problem, but now reward is what the discriminator says. 
    states = np.vstack(memory[:, 0]) 
    actions = list(memory[:, 1]) 
    rewards = list(memory[:, 2])  #IRL Rewards
    masks = list(memory[:, 3]) 

    # compute value of what happened, see if what we can get us better. 
    old_values = critic(torch.Tensor(states))
    #GAE aka estimate of Value + actual return roughtly 
    returns, advants = get_gae(rewards, masks, old_values, gamma, lamda)
    
    # pass states through actor, get corresponding actions
    mu, std = actor(torch.Tensor(states))
    # new mus and stds? 
    old_policy = log_prob_density(torch.Tensor(actions), mu, std) # sum of log probability
    # of old actions

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(actor_critic_update_num):
        np.random.shuffle(arr)

        for i in range(n // batch_size): 
            batch_index = arr[batch_size * i : batch_size * (i + 1)]
            #batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
        
        
            values = critic(inputs) #
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -clip_param, 
                                         clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - clip_param,
                                        1.0 + clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()
            #print(actor_loss,critic_loss,entropy)
           # return actor_loss, critic_loss, entropy
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy #entropy bonus to promote exploration.

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()

           # critic_optim.zero_grad()
           # loss.backward() 
            critic_optim.step()

def get_gae(rewards, masks, values, gamma, lamda):
    """
    How much better a particular action is in a particular state. 
    
    Uses reward of current action + value function of that state-action pair, discount factor gamma, and then lamda to compute. 
    """
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))): #for LL, only ever one step :-)
        running_returns = rewards[t] + (gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        running_advants = running_delta + (gamma * lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    """
    The loss for PPO. Re-run through network, recomput policy from states
    and see if this surrogate ratio is better. If it is, use as proximal policy update. It's very close to prior policy, but def better. 
    
    Not sure this actually works though. Should not the new mu and stds be used to draw,
    
        When do we use get_action? Only once in main, I think it should be for all? 
    """
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy