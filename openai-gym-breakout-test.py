import gym
env = gym.make('Breakout-v0')
env.reset()
for i_episode in range(1000): #initially, i_episode was _
    env.render()
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample() # take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        print '-----t-------'
    print '#########'
