# limitation-learning

This repository houses the code for the NYU Deep Reinforcement Learning Fall 2020 Final Project by Rahul Zalkikar and Noah Kasmanoff, "Limitation Learning: Probing the Behavior of Large Language Models with Imitation Learning". 



## Introduction

The exponential popularity and implementation of artificial intelligence and machine learning owes much credit to the era "big data", and the  of crafting algorithms with millions (and now trillions!) of parameters. This has allowed for the onset and implementation of deep learning models to natural language processing, with industrial applications such as conversational chat-bots. These chat-bots are often specialized use cases of large language models, and not easily understandable. The purpose of this work is to demonstrate the possible application of imitation learning and inverse reinforcement learning as a means to characterize such language models, and mitigate the negative consequences of adverse dialog. 


We apply generative adversarial imitation learning (GAIL) to produce a proxy for the reward function present in a basic conversation, using the Cornell Movie Dialog Corpus. We apply imitation learning to craft coherent replies to the input utterance. 

Down the line, our focus is on an auxilary objective of GAIL, using a discriminator network as a proxy for a reward function.

We hope that after training the policy and discriminator networks to equilibrium, we may use this proxy reward function as a way to probe black box language models with direct feedback. We can then additionally feed inputs to publicly available conversational AI, extract a response, and pass this through the proxy reward to gain a better intuition on that language model's behavior. 

This work is just the beginning. We emphasize that GAIL is a method of imitation, not inverse reinforcement learning. This distinction is important in that we cannot recover the underlying reward function of the system, but instead a proxy based on imitation. We consider the application of more advanced techniques such as guided cost learning a worthwhile next step if GAIL succeeds.

## Methods

Data. For this work, we use conversations from the Cornell-Movie-Dialog corpus, which contains over xxx conversations. We first split this corpus into conversation pairs, and use those pairs for training. In terms of a reinforcement learning algorithm, we still aim to maximize an expected reward over a trajectory, however in all cases the trajectory, or conversation length, is 1. It is possible to extend this technique to conversations of varying length, but for the purposes of proof-of-concept we do not explore this approach. 

As a means to accelerate the development of this work, we use Spacy and it's embedding software to quickly transform the words/tokens of our dataset into vector embeddings. We should note this introduces some bias to our dataset, seeing as the embeddings originate from an outside source, Google News, but we attempt to mitigate this by fine-tuning our results. This phase of the project transforms our conversations into a vectorized, model-readable form which is then used for training. 


Generative Adversarial Imitiation Learning. GAIL casts the objective of imitation learning, into a min max optimization problem analagous to generative adversarial networks.

GAIL uses a discriminator network, characterized by loss function $E_{\pi}[log(D(s,a))] + E_{\pi_E}[log(1-D(s,a))]$ to distinguish between policy and expert generated actions given a state input, in the process creating a proxy for the reward function ubiquitous to RL, formulated as $E_{\tau}[\nabla_{\theta} log(\pi_{\theta}(a | s) r(s,a)] - \lambda \nabla_{\theta} H(\pi_{\theta}) $ where $ r(s,a) = -log(D(s,a))$ 


Using this reward function, a policy is trained through standard model-free reinforcement learning technqiues to maximize this reward. If successful, the policy is able to "trick" the discriminator into classifying it's trajectory as an expert demonstration, and the game between the networks continue. 

In this work, we  deploy this technique in the context of dialog. 




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

By training GAIL, we achieve a policy capable of producing resonable responses to common utterances from movie dialog. Initializing our word embeddings from y. 

Additionally, we achieve a discriminator which serves as a proxy reward function for state-action pairs. Using this proxy reward function, we examine the topology of state-action pairs, and examine how an out of the box language model performs as a result of our proxy. 

This work is an important first step in better characterizing large language models from an outside perspective. By casting dialog as a reinforcement learning problem, we are able to acquire direct feedback in the form of a reward function which indicates how similar state-action pairs are received. 


## Next Steps

In future work, we hope to extend our task to extended conversations, acquire access to GPT3, and take a step farther than imitation and use inverse reinforcement learning as a more accurate representation of dialog reward dynamics. 



