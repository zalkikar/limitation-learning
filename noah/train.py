import numpy as np
import sys,os
import torch
import torch.nn as nn
import data as data
import architecture as architecture

####################################### INPUT ##########################################

## BC Pre-train

## Discriminator Training GAIL But mostly getting better at Discrim
 
## You can estimate reward of arbitrary inputs i.e GPT, or us. Discussion

## Conclusion of 

realizations = 1000
mode         = 'train'
sim_name     = 'IllustrisTNG'
seed         = 1
z            = 5.00
batch_size   = 32
epochs       = 1000

# optimizer parameter
learning_rate = 1e-3
weight_decay  = 1e-3
fout          = 'results.txt'

# best-model fname
f_best_model = 'BestModel_model6_%s_%.1e_%.1e_%d_z=%.2f.pt'\
               %(sim_name, learning_rate, weight_decay, batch_size, z)
########################################################################################

# create the training, validation and testing datasets
train_loader, valid_loader, test_loader = data.create_datasets(sim_name, seed, 
                                            realizations, z, batch_size)

# define the architecture
model = architecture.model6(1,16,16,32,32)
total_params = sum(p.numel() for p in model.parameters())
print 'Total number of parameters in the network: %d'%total_params

# define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, 
                                                       patience=10, verbose=True)

# load best-model in case it exists
if os.path.exists(f_best_model):  
    print 'loading best model...'
    model.load_state_dict(torch.load(f_best_model))

# do validation with the best-model and compute loss
model.eval() 
count, best_loss = 0, 0.0
with torch.no_grad():
    for maps, params_true in valid_loader:
        params_valid = model(maps)
        error    = criterion(params_valid, params_true)
        best_loss += error.numpy()
        count += 1
best_loss /= count
print 'validation error = %.3e'%best_loss

# main loop: train and validate

### BEHAVIORAL CLONING


if os.path.exists(fout):  os.system('rm %s'%fout)
for epoch in range(epochs):

    # TRAIN
    model.train()
    count, loss_train = 0, 0.0
    for maps, params_true in train_loader: #REPLACE WITH WHAT YOU HAVE
 
        # Forward Pass
        optimizer.zero_grad()
        params_pred = model(maps)
        loss    = criterion(params_pred, params_true)
        loss_train += loss.detach().numpy()
        
        # Backward Prop
        loss.backward()
        optimizer.step()
        
        count += 1
    loss_train /= count
    
    # VALID
    model.eval() 
    count, loss_valid = 0, 0.0
    with torch.no_grad():
        for maps, params_true in valid_loader:
            params_pred = model(maps)
            error    = criterion(params_pred, params_true)   
            loss_valid += error.numpy() # WHY NOT SIMILARITY TOO!>!>!>!
            count += 1
    loss_valid /= count
     
    # TEST
    model.eval() 
    count, loss_test = 0, 0.0
    with torch.no_grad():
        for maps, params_true in test_loader:
            params_pred = model(maps)
            error    = criterion(params_pred, params_true) 
            loss_test += error.numpy()
            count += 1
    loss_test /= count
    
    # Save Best Model  #CAN DO IT BASED ON SIMILARITY OR JUST LOSS OR EVERYTHING WHO KNOWS
    if loss_valid<best_loss:
        best_loss = loss_valid
        torch.save(model.state_dict(), f_best_model)
        print('%03d %.4e %.4e %.4e (saving)'\
              %(epoch, loss_train, loss_valid, loss_test))
    else:
        print('%03d %.4e %.4e %.4e'%(epoch, loss_train, loss_valid, loss_test)) # 
    
    # update learning rate
    scheduler.step(loss_valid)

    # save results to file
    f = open(fout, 'a') # THIS IS WHAT ALLOWS YOU TO SEE RESULTS IN REAL TIME> MAYBE ADD SIMILAIRTY AND DROP TEST?
    f.write('%d %.4e %.4e %.4e\n'%(epoch, loss_train, loss_valid, loss_test))
    f.close()