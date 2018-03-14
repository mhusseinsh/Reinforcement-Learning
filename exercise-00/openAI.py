import gym
env = gym.make('CartPole-v0')
highscore = 0
for i_episode in range(20): # run 20 episodes
    observation = env.reset()
    points = 0 # keep track of the reward each episode
    for t in range(100): # run for 10 timesteps or until done, whichever is first
        env.render()
        print("Observations for the", i_episode, " episode and ", t, "timestep:")
        print(observation)
        # Simple feedback control loop, change action according to current observation (angle)
        action = 1 if observation[2] > 0 else 0 # if angle is positive, move right. Else, move left
        observation, reward, done, info = env.step(action)
        print("Reward: ", reward)
        points += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break