# limitation-learning

This repository houses the code for the NYU Deep Reinforcement Learning Fall 2020 Final Project by Rahul Zalkikar and Noah Kasmanoff, "Limitation Learning: Capturing Adverse Dialog with GAIL". 




We apply generative adversarial imitation learning (GAIL) to produce a proxy for the reward function present in a basic conversation, using the Cornell Movie Dialog Corpus. We apply imitation learning to craft coherent replies to the input utterance. 

Down the line, our focus is on an auxilary objective of GAIL, using a discriminator network as a proxy for a reward function.

We hope that after training the policy and discriminator networks to equilibrium, we may use this proxy reward function as a way to probe black box language models with direct feedback. We can then additionally feed inputs to publicly available conversational AI, extract a response, and pass this through the proxy reward to gain a better intuition on that language model's behavior. 

This work is just the beginning. We emphasize that GAIL is a method of imitation, not inverse reinforcement learning. This distinction is important in that we cannot recover the underlying reward function of the system, but instead a proxy based on imitation. We consider the application of more advanced techniques such as guided cost learning a worthwhile next step if GAIL succeeds.
