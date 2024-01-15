import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN import *
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


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
    print 'pretrained episode = {}'.format(episode)
    print 'pretrained epsilon = {}'.format(epsilon)
    model.load_state_dict(checkpoint['state_dict'])
    time_step = checkpoint.get('best_time_step', None)
    if time_step is None:
        time_step = checkpoint('time_step')
    print 'pretrained time step = {}'.format(time_step)
    return episode, epsilon, time_step

def train_dqn(model, options, resume):
    """Train DQN
    model -- DQN model
    lr -- learning rate
    max_episode -- maximum episode
    resume -- resume previous model
    model_name -- checkpoint file name
    """
    best_time_step = 0.
    if resume:
        if options.weight is None:
            print 'when resume, you should give weight file name.'
            return
        print 'load previous model weight: {}'.format(options.weight)
        _, _, best_time_step = load_checkpoint(options.weight, model)

    flappyBird = game.GameState()
    optimizer = optim.Adam(model.parameters(), lr=options.lr)
    ceriterion = nn.MSELoss()

    action = [1, 0, 0, 0]
    state_0 = np.zeros((1000 + 2 + 1 + 2 * 20))
    state, r, terminal = flappyBird.step(action, state_0)
    model.set_initial_state()

    if options.cuda:
        model = model.cuda()
    # in the first `OBSERVE` time steos, we dont train the model
    for i in range(options.observation):
        action = model.get_action_randomly()
        state_next, r, terminal = flappyBird.step(action, state)
        model.store_transition(state_next, action, r, terminal)
        state = state_next
    # start training
    for episode in range(options.max_episode):
        model.time_step = 0
        model.set_train()
        total_reward = 0.
        state= np.zeros((1000 + 2 + 1 + 2 * 20))
        # begin an episode!
        while True:
            optimizer.zero_grad()
            action = model.get_action()
            state_next, r, terminal = flappyBird.step(action, state)
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
            max_q, _ = torch.max(q_value_next, dim=1)

            for i in range(options.batch_size):
                if not minibatch[i][4]:
                    y[i] += options.gamma*max_q.data[i][0]

            y = Variable(torch.from_numpy(y))
            action_batch_var = Variable(torch.from_numpy(action_batch))
            if options.cuda:
                y = y.cuda()
                action_batch_var = action_batch_var.cuda()
            q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)

            loss = ceriterion(q_value, y)
            loss.backward()

            optimizer.step()
            # when the bird dies, the episode ends
            if terminal:
                break

        print 'episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
                episode, model.epsilon, model.time_step, total_reward)

        if model.epsilon > options.final_e:
            delta = (options.init_e - options.final_e)/options.exploration
            model.epsilon -= delta

        if episode % 100 == 0:
            ave_time = test_dqn(model, episode)

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
        print 'save checkpoint, episode={}, ave time step={:.2f}'.format(
                 episode, ave_time)

def test_dqn(model, episode):
    """Test the behavor of dqn when training

       model -- dqn model
       episode -- current training episode
    """
    model.set_eval()
    ave_time = 0.
    for test_case in range(5):
        model.time_step = 0
        state= np.zeros((1000 + 2 + 1 + 2 * 20))
        flappyBird = game.GameState()
        state, r, terminal = flappyBird.step([1, 0, 0, 0], state)
        model.set_initial_state()
        while True:
            action = model.get_optim_action()
            state_next, r, terminal = flappyBird.step(action, state)
            state = state_next
            if terminal:
                break
            model.current_state = state_next
            model.increase_time_step()
        ave_time += model.time_step
    ave_time /= 5
    print 'testing: episode: {}, average time: {}'.format(episode, ave_time)
    return ave_time


def play_game(model_file_name, cuda=False, best=True):
    """Play flappy bird with pretrained dqn model

       weight -- model file name containing weight of dqn
       best -- if the model is best or not
    """
    print 'load pretrained model file: ' + model_file_name
    model = BrainDQN(epsilon=0., mem_size=0, cuda=cuda)
    load_checkpoint(model_file_name, model)

    model.set_eval()
    bird_game = game.GameState()
    model.set_initial_state()
    state= np.zeros((1000 + 2 + 1 + 2 * 20))
    if cuda:
        model = model.cuda()
    while True:
        action = model.get_optim_action()
        state_next, r, terminal = bird_game.step(action, state)
        state = state_next
        if terminal:
            break

        model.current_state = state_next
        model.increase_time_step()
    print 'total time step is {}'.format(model.time_step)
