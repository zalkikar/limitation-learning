# limitation-learning

This repository houses the code for the NYU Deep Reinforcement Learning Fall 2020 Final Project by Rahul Zalkikar and Noah Kasmanoff, "Limitation Learning: Probing the Behavior of Large Language Models with Imitation Learning". 

In this work, we apply generative adversarial imitation learning (GAIL) to produce a proxy for the reward function present in a basic conversation, using data pulled from the Cornell Movie Corpus dataset (link). Our purpose is to show that by using GAIL, we can use imitation learning to craft an agent capable of formulating coherent responses, or actions, to the input utterance, or state. 

In particular, our focus is on an auxilary goal of GAIL, which is using a discriminator network as a proxy for a reward function that central to reinforcement learning. For more information on how exactly this reward function operates, please refer to our methodology and background. 

This proxy reward function is the crux of our contribution. We hope that after training the policy and discriminator networks to equilibrium, we may use this proxy reward function as a way to probe black box language models with direct feedback. Essentially given a state utterance and action utterance, our reward function allows the user to see how high or low this pair is, in comparison to similar state action pairs.

We feed inputs to conversational AI, extract responses, and pass this through the reward function to gain a better intuition that language model's performance. 


This work is just the beginning 


## Introduction

## Methods

Data. For this work, we use conversations from the Cornell-Movie-Dialog corpus, which contains over xxx conversations. We first split this corpus into conversation pairs, and use those pairs for training. In terms of a reinforcement learning algorithm, we still aim to maximize an expected reward over a trajectory, however in all cases the trajectory, or conversation length, is 1. It is possible to extend this technique to conversations of varying length, but for the purposes of proof-of-concept we do not explore this approach. 

As a means to accelerate the development of this work, we use Spacy and it's embedding software to quickly transform the words/tokens of our dataset into vector embeddings. We should note this introduces some bias to our dataset, seeing as the embeddings originate from an outside source, Google News, but we attempt to mitigate this by fine-tuning our results. This phase of the project transforms our conversations into a vectorized, model-readable form which is then used for training. 


GAIL.
GAIL is a powerful technique for imitation/inverse reinforcement learning. While there are many ways to extract the intuition from it, the key insight we draw from GAIL's formulations is that ....

TODO
...



Networks. 
After discussing the specifics of GAIL, we now touch on the various architectures employed. 

For the policy network, we use a sequence-to-sequence architecture, which maps the input state, a matrix of size Nx300 where (N represents the number of tokens in the sequence, and 300 represents that each token is a 300 dimensional vector) to a similarly sized matrix of Nx300. 

Of this output, each element now corresponds to the mean of a gaussian distribution with standard deviation Y. This is to encourage exploration, while simultaneously allowing this to obey the policy gradient formulation, albeit in a slightly more sophisticated manner. 

TODO


Training:

For training we follow the method laid out in the original GAIL paper, with a few changes (given the fixed trajectory length of 1), and reflect those differences in the algorithm described: 


In our work, it is not straightforward how we would determine stopping criteria. 
Since we do not have access to an underlying reward function aside from our proxy, it is impossible to determine how close or not our training has gotten to achieving equilibrium between the networks. 

As an alternative, we draw conclusions on GAIL success by examining the resulting actions the policy selects. As an example of this evolution and determination to stop training, we present the following actions selected by the network at different phases of training:

Phase 1: Action  = asdlk;fa;lisdfjkl;as;dkf
Phase 21241: Action = I love you dear, how about a pizza. 

Once this network has met our standards, we extract the discriminator and use it's log to extract rewards for different samples. 



## Results

GAIL has a variety of hyper-parameters to tune. By choosing our trajectories to be a single timestep, we successfully reduce this by eliminating values such as discount factor gamma, and GAE factor lambda. Even so, we still faced many values to tune. To do so we train our model on NYU Greene using a V100 GPU, and log on tensorboard the corresponding average expert accuracies, example policy state action pairs (in comparison to their expert companions), and finally evaluate the reward of random actions and large language model actions. 



## Conclusion


## Next Steps
