"""
Main loop for training GAIL. 

TODO: Add tensorboard logging tools, where to save models, and then pointers to analysis scripts. 
"""

from gail_utils import * #this should be everything


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc = EncRnn(hidden_size=64, num_layers=2, embed_size=EMBED_DIM)
dec = DecRnn(hidden_size=64, num_layers=2, embed_size=EMBED_DIM, output_size=VOCAB_SIZE)
enc_ = EncRnn(hidden_size=64, num_layers=2, embed_size=EMBED_DIM)
dec_ = DecRnn(hidden_size=64, num_layers=2, embed_size=EMBED_DIM, output_size=VOCAB_SIZE)
agent = Seq2SeqAttn(enc, dec, TRG_PAD_IDX, VOCAB_SIZE, device).to(device)
# load in pre-trained model 
agent_ = Seq2SeqAttn(enc_, dec_, TRG_PAD_IDX, VOCAB_SIZE, device).to(device)
agent.load_state_dict(torch.load(
    '/scratch/nsk367/deepRL/limitation-learning/src/pretrained_generators/model-epoch10.pt'))
agent_.load_state_dict(torch.load(
    '/scratch/nsk367/deepRL/limitation-learning/src/pretrained_generators/model-epoch10.pt'))


discrim = Discriminator(model=agent_,SEQ_LEN=5).to(device)


policy_optim = optim.Adam(agent.parameters(), lr=args.agent_learning_rate)

discrim_optim = optim.Adam(discrim.parameters(), lr=args.discrim_learning_rate)


train_discrim_flag = True
train_policy_flag = True
policy_losses = []
discrim_losses = []
expert_accs = []
learner_accs = []

for step in range(args.max_iter_num):
    score = 0
    discrim_memory = deque()
    learner_memory = deque()
    trajectories = random.sample(d.items(),k=args.batch_size)
    for (index, vects) in trajectories:
        input_state, expert_action = vects
        input_state = torch.cat((torch.LongTensor([sos_ind]), 
                                 input_state,
                                 torch.LongTensor([eos_ind])), 
                                 dim=0).to(device)
        expert_action = torch.cat((torch.LongTensor([sos_ind]), 
                                expert_action, 
                                torch.LongTensor([eos_ind])), 
                               dim=0).to(device)
        action_probs =  get_action_probs(agent, input_state, sos_ind, eos_ind, SEQ_LEN, device)
        action, action_log_probs = get_action(action_probs)
       # print(action)]
        action = action.detach()
        action = pad_action(action)
        #print(action)
    
        #obtain what the estimated reward is for this 
        irl_reward = get_reward(discrim,input_state,action)
        score += irl_reward
        learner = discrim(input_state.unsqueeze(0).detach(), action.unsqueeze(0).detach()) #pass (s,a) through discriminator
        expert = discrim(input_state.unsqueeze(0), expert_action.unsqueeze(0)) #pass (s,a) through discriminator

        discrim_memory.append([learner,expert])
        learner_memory.append([-irl_reward  * action_log_probs])
    if train_policy_flag:
        policy_loss = train_policy(learner_memory, policy_optim, args)   
        policy_losses.append(policy_loss)
    print(score)
    if train_discrim_flag:
        discrim_loss, expert_acc, learner_acc = train_discrim(discrim_memory, discrim_optim, args) 
        print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
        discrim_losses.append(discrim_loss)
        expert_accs.append(expert_acc)
        learner_accs.append(learner_acc)


      #  writer.add_scalar('log/expert_acc', float(expert_acc), iter) #logg
      #  writer.add_scalar('log/learner_acc', float(learner_acc), iter) #logg
      #  writer.add_scalar('log/avg_acc', float(learner_acc + expert_acc)/2, iter) #logg
        if args.suspend_accu_exp is not None: #only if not None do we check.
            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False
  
    expert_tokens = " ".join([words[int(ind)] for ind in expert_action])
    action_tokens = " ".join([words[int(ind)] for ind in action])
    state_tokens = " ".join([words[int(ind)] for ind in input_state])
    print(state_tokens,'|',action_tokens,'|',expert_tokens)

