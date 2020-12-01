import math
import torch
from torch.distributions import Normal, Categorical

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def get_discrete_action(action_probs):
    
    dist = Categorical(action_probs)
    action = dist.sample()
    
    action_log_prob = dist.log_prob(action)
    
    return action, action_log_prob

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

    Key for project. 
    
    Reward is higher the closer this is to 0, because the more similar it is to an expert action. :
    Is quite close to imitation learning, but hope here is that with such a large number of expert demonstrations and entropy bonuses etc. it learns more than direct imitation. 
    """
    state = torch.Tensor(state)
    try:
        action = torch.Tensor(action)
    except:
        pass
    state_action = torch.cat([state, action])
    with torch.no_grad():
        return -math.log(discrim(state_action)[0].item())

def save_checkpoint(state, filename):
    torch.save(state, filename)