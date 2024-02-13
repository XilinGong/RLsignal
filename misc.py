import sys
sys.path.append("game/")
import environment as game
from BrainDQN import *
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import warnings
import time
import matplotlib.pyplot as plt
import csv
warnings.filterwarnings('ignore')
import numpy as np

datasize = 1000
test_datasize = 2000


def load_data_file(index):
    data_file = r'/home/stat/Xilin/RL_simu/simu_20000_0.1_90_140_train.npy'
    data_set = np.load(data_file)  # (10 seconds * 100Hz) + ID + Time + H + R + S + D
    num_labels = 6
    signal = data_set[:datasize, 0:-num_labels]
    label_s = data_set[:, -2]
    label_d = data_set[:, -1]
    return signal[index], label_s[index], label_d[index]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint model to disk

        state -- checkpoint state: model weight and other info
                binding by user
        is_best -- if the checkpoint is the best. If it is, then
                save as a best model
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(filename, model):
    """Load previous checkpoint model

    filename -- model file name
    model -- DQN model
    """
    try:
        checkpoint = torch.load(filename)
    except:
        # load weight saved on gpy device to cpu device
        # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(filename, map_location=lambda storage, loc:storage)
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    print ('pretrained episode = {}'.format(episode))
    print ('pretrained epsilon = {}'.format(epsilon))
    model.load_state_dict(checkpoint['state_dict'])
    time_step = checkpoint.get('best_time_step', None)
    if time_step is None:
        time_step = checkpoint('time_step')
    print ('pretrained time step = {}'.format(time_step))
    return episode, epsilon, time_step

def train_dqn(model, options, resume):
    data_file = r'/home/stat/Xilin/RL_simu/simu_20000_0.1_90_140_train.npy'
    data_set = np.load(data_file)  # (10 seconds * 100Hz) + ID + Time + H + R + S + D
    num_labels = 6
    signal_dataset = data_set[:, 0:-num_labels]
    label_s_dataset = data_set[:, -2]
    label_d_dataset = data_set[:, -1]
    q_loss = []
    ave_reward_list = []
    ave_timestep_list = []

    best_time_step = 0.
    if resume:
        if options.weight is None:
            print ('when resume, you should give weight file name.')
            return
        print ('load previous model weight: {}'.format(options.weight))
        _, _, best_time_step = load_checkpoint(options.weight, model)

    # for index in range(datasize-1)
    index =random.randint(0, datasize-1)
    
    agent = game.GameState(label_s=label_s_dataset[index], label_d=label_d_dataset[index], label_rad=2)
    agent.players, agent.playerd = 115.0, 80.0
    optimizer = optim.Adam(model.parameters(), lr=options.lr)
    ceriterion = nn.MSELoss()

    cur_signal = signal_dataset[index]
    mid_s = (max(label_s_dataset)+min(label_s_dataset))/2
    mid_d = (max(label_d_dataset)+min(label_d_dataset))/2
    cur_rad = max((max(label_s_dataset)-min(label_s_dataset))/2, (max(label_d_dataset)-min(label_d_dataset))/2)
    state_0 = np.concatenate((cur_signal,np.array([mid_s, mid_d, cur_rad])))
    all_zeros = np.zeros(1000 + 2 + 1 + 4 * 5)
    all_zeros[:len(state_0)] = state_0
    state_0 = all_zeros
    model.set_initial_state(state_0)
    action = model.get_action_randomly()
    state, r, terminal = agent.step(action, state_0)


    if options.cuda:
        model = model.cuda()
    # in the first `OBSERVE` time steos, we dont train the model
    
    for i in range(options.observation):
        index =random.randint(0, datasize-1)

        agent = game.GameState(label_s=label_s_dataset[index], label_d=label_d_dataset[index], label_rad=2)
        agent.players, agent.playerd = 115.0, 80.0
        optimizer = optim.Adam(model.parameters(), lr=options.lr)
        ceriterion = nn.MSELoss()

        cur_signal = signal_dataset[index]
        mid_s = (max(label_s_dataset)+min(label_s_dataset))/2
        mid_d = (max(label_d_dataset)+min(label_d_dataset))/2
        cur_rad = max((max(label_s_dataset)-min(label_s_dataset))/2, (max(label_d_dataset)-min(label_d_dataset))/2)
        state_0 = np.concatenate((cur_signal,np.array([mid_s, mid_d, cur_rad])))
        all_zeros = np.zeros(1000 + 2 + 1 + 4 * 5)
        all_zeros[:len(state_0)] = state_0
        state_0 = all_zeros
        model.set_initial_state(state_0)
        
        action = model.get_action_randomly()
        state, r, terminal = agent.step(action, state_0)
        action = model.get_action_randomly()
        state_next, r, terminal = agent.step(action, state)
        model.store_transition(state_next, action, r, terminal)
        state = state_next
        if terminal:
            index =random.randint(0, datasize-1)
            cur_signal = signal_dataset[index]
            state = np.concatenate((cur_signal,np.array([mid_s, mid_d, cur_rad])))
            all_zeros = np.zeros(1000 + 2 + 1 + 4 * 5)
            all_zeros[:len(state)] = state
            state = all_zeros
            agent.label_s = label_s_dataset[index]
            agent.label_d = label_d_dataset[index]
            agent.rad = cur_rad
            agent.players, agent.playerd = 115.0, 80.0
            model.set_initial_state(state)
            
    print('startttttttttttttttttttingggggggggg')
    # start training
    for episode in range(options.max_episode):
        for i in range(datasize-1):
            index =random.randint(0, datasize-1)
            model.time_step = 0
            model.set_train()
            total_reward = 0.
            cur_signal = signal_dataset[index]
            agent.label_s = label_s_dataset[index]
            agent.label_d = label_d_dataset[index]
            agent.rad = cur_rad
            agent.players, agent.playerd = 115.0, 80.0
            state = np.concatenate((cur_signal,np.array([mid_s, mid_d, cur_rad])))
            all_zeros = np.zeros(1000 + 2 + 1 + 4 * 5)
            all_zeros[:len(state)] = state
            state = all_zeros
            model.set_initial_state(state)
            # begin an step!
            while True:
                optimizer.zero_grad()
                action = model.get_action()

                state_next, r, terminal = agent.step(action, state, True)
                total_reward += options.gamma**model.time_step * r
                model.store_transition(state_next, action, r, terminal)
                model.increase_time_step()
                state = state_next

                # Step 1: obtain random minibatch from replay memory
                minibatch = random.sample(model.replay_memory, options.batch_size)
                state_batch = np.array([data[0] for data in minibatch])
                action_batch = np.array([data[1] for data in minibatch])   
                reward_batch = np.array([data[2] for data in minibatch])              
                next_state_batch = np.array([data[3] for data in minibatch])

                state_batch_var = Variable(torch.from_numpy(state_batch))
                next_state_batch_var = Variable(torch.from_numpy(next_state_batch),
                                            volatile=True)
                if options.cuda:
                    state_batch_var = state_batch_var.cuda()
                    next_state_batch_var = next_state_batch_var.cuda()
                # Step 2: calculate y
                q_value_next = model.forward(next_state_batch_var)
                q_value = model.forward(state_batch_var)
                y = reward_batch.astype(np.float32)
                max_q, max_q_2 = torch.max(q_value_next, dim=1)
                print(y)
                for i in range(options.batch_size):
                    if not minibatch[i][4]:
                        y[i] += options.gamma*max_q.data[i]
                print(y)
                y = Variable(torch.from_numpy(y))
                action_batch_var = Variable(torch.from_numpy(action_batch))

                if options.cuda:
                    y = y.cuda()
                    action_batch_var = action_batch_var.cuda()
                q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)

                loss = ceriterion(q_value, y)
                loss.backward()

                q_loss.append(loss.item())
                optimizer.step()
                if terminal:
                    break


        # print ('episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
        #         episode, model.epsilon, model.time_step, total_reward))

        if model.epsilon > options.final_e:
            delta = (options.init_e - options.final_e)/options.exploration
            model.epsilon -= delta

        if episode % 10 == 0:
            ave_time, ave_reward = test_dqn(model, episode, index)
            ave_reward_list.append(ave_reward)
            ave_timestep_list.append(ave_time)

        if len(ave_reward_list)% 5 ==0:
            plt.plot([x*10 for x in range(len(ave_reward_list))], ave_reward_list)
            plt.xlabel('epoch')
            plt.ylabel('test reward')
            plt.savefig('reward'+str(len(ave_reward_list))+'.png')
            plt.close()

            plt.plot([x*10 for x in range(len(ave_timestep_list))], ave_timestep_list)
            plt.xlabel('epoch')
            plt.ylabel('test timestep')
            plt.savefig('timestep'+str(len(ave_reward_list))+'.png')
            plt.close()

        if ave_time > best_time_step:
            best_time_step = ave_time
            save_checkpoint({
                'episode': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'best_time_step': best_time_step,
                }, True, 'checkpoint-episode-%d.pth.tar' %episode)
        elif episode % options.save_checkpoint_freq == 0:
            save_checkpoint({
                'episode:': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'time_step': ave_time,
                }, False, 'checkpoint-episode-%d.pth.tar' %episode)
        else:
            continue
        print ('save checkpoint, episode={}, ave time step={:.2f}'.format(
                episode, ave_time))

def test_dqn(model, episode, index):
    data_file = r'/home/stat/Xilin/RL_simu/simu_20000_0.1_90_140_train.npy'
    data_set = np.load(data_file)  # (10 seconds * 100Hz) + ID + Time + H + R + S + D
    num_labels = 6
    signal_dataset = data_set[:, 0:-num_labels]
    label_s_dataset = data_set[:, -2]
    label_d_dataset = data_set[:, -1]
    mid_s = (max(label_s_dataset)+min(label_s_dataset))/2
    mid_d = (max(label_d_dataset)+min(label_d_dataset))/2
    cur_rad = max((max(label_s_dataset)-min(label_s_dataset))/2, (max(label_d_dataset)-min(label_d_dataset))/2)

    """Test the behavor of dqn when training

       model -- dqn model
       episode -- current training episode
    """
    model.set_eval()
    ave_time = 0.
    ave_reward = 0.
    testtime = 100
    for test_case in range(testtime):
        model.time_step = 0
        
        index =random.randint(0, datasize-1)
        cur_signal = signal_dataset[index]
        state = np.concatenate((cur_signal,np.array([mid_s, mid_d, cur_rad])))
        all_zeros = np.zeros(1000 + 2 + 1 + 4 * 5)
        all_zeros[:len(state)] = state
        state = all_zeros
        agent = game.GameState(label_s_dataset[index], label_d_dataset[index], 2)
        agent.players, agent.playerd = 115.0, 80.0
        agent.rad = cur_rad
        model.set_initial_state(state)
        # action0 = model.get_optim_action()
        # state, r, terminal = agent.step(action0, state)
        
        while True:
            action = model.get_optim_action()
            state_next, r, terminal = agent.step(action, state, True)
            state = state_next
            model.increase_time_step()
            ave_reward += r
            if terminal:
                break
            model.current_state = state_next
        ave_time += model.time_step
    ave_time /= testtime
    ave_reward /= testtime
    print ('testing: episode: {}, average time: {}, average reward:{}'.format(episode, ave_time, ave_reward))
    return ave_time, ave_reward