import gym
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers, activations
import numpy as np
import matplotlib.pyplot as plt
import data_file


try:
    tf.enable_eager_execution()

except:
    pass

class unit_sphere_projection_layer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(unit_sphere_projection_layer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    pass


  def call(self, inputs):
    return tf.math.l2_normalize(inputs)

class DDPG:

    def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0, polyak_factor=0, buff_size=1000, samp_size=64):
        #num_veh = 11
        self.num_states = data_file.num_features*data_file.num_veh#data_file.max_vehi_per_lane*data_file.lane_max#env.observation_space.shape[0]
        self.num_actions = data_file.num_veh 
        self.noise_std_dev = noise_std_dev

        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))

        self.actor_model_ = self.get_actor()
        self.critic_model_ = self.get_critic()

        self.target_actor_ = self.get_actor()
        self.target_critic_ = self.get_critic()

        # Making the weights equal initially
        self.target_actor_.set_weights(self.actor_model_.get_weights())
        self.target_critic_.set_weights(self.critic_model_.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = cri_lr
        self.actor_lr = act_lr

        self.critic_optimizer_ = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer_ = tf.keras.optimizers.Adam(self.actor_lr)

        # Discount factor for future rewards
        self.gamma_ = disc_factor
        # Used to update target networks
        self.tau_ = polyak_factor

        self.buff_size = buff_size
        self.samp_size = samp_size

        self.buffer = Buffer(buffer_capacity=self.buff_size, batch_size=self.samp_size, state_size=self.num_states, action_size=self.num_actions, actor_m=self.actor_model_, critic_m=self.critic_model_, tar_act_m=self.target_actor_, tar_cri_m=self.target_critic_, gamma=self.gamma_, tau=self.tau_, cri_optimizer=self.critic_optimizer_, act_optimizer=self.actor_optimizer_)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,1))

        layer_1 = layers.Flatten() (inputs)

        layer_1 = layers.Dense(4*self.num_actions, activation='relu') (layer_1)

        layer_2 = layers.Dense(2*self.num_actions) (layer_1)

        layer_3 = layers.Dense(self.num_actions) (layer_2)


        '''# bn_inputs = layers.BatchNormalization() (inputs)

        #inputs = layers.Reshape(target_shape=(self.num_states,1)) (inputs)

        layer_1 = layers.Conv1D(4, data_file.num_features, strides=data_file.num_features, activation="relu") (inputs)

        layer_1 = layers.Flatten() (layer_1)

        layer_1 = layers.Reshape(target_shape=(layer_1.shape[-1], 1)) (layer_1)

        layer_2 = layers.Conv1D(2, kernel_size=4, strides=4) (layer_1)

        layer_2 = layers.Flatten() (layer_2)

        layer_2 = layers.Reshape(target_shape=(layer_2.shape[-1], 1)) (layer_2)

        layer_3 = layers.Conv1D(1, 2, strides=2, kernel_initializer=last_init, bias_initializer=last_init) (layer_2)

        layer_3 = layers.Flatten() (layer_3)'''

        outputs_pi = activations.softmax(layer_3)
        
        model = tf.keras.Model(inputs, outputs_pi)
        return model


    def get_critic(self):

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # State as input
        state_input = layers.Input(shape=(self.num_states))
        bn_state_input = layers.BatchNormalization() (state_input)


        state_out = layers.Dense(256, activation="relu")(bn_state_input)
        bn_state_input = layers.BatchNormalization() (state_out)

        state_out = layers.Dense(64, activation="relu")(bn_state_input)
        bn_state_input = layers.BatchNormalization() (state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(64, activation="relu")(action_input)
        bn_action_out = layers.BatchNormalization() (action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([bn_state_input, bn_action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        critic_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], critic_out)

        return model


    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model_(state))
        noise = noise_object()

        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise
        
        # We make sure action is within bounds
        legal_action = sampled_actions 

        return [np.squeeze(legal_action)]

    

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=5e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.t = 0
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        self.t += self.dt
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.random.normal(size=self.mean.shape) * (1/self.t)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, state_size=12, action_size=5, actor_m=None, critic_m=None, tar_act_m=None, tar_cri_m=None, gamma=0.99, tau=0.001, cri_optimizer=None, act_optimizer=None):
        # Number of "experiences" to store at max
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.actor_model = actor_m
        self.critic_model = critic_m
        self.target_actor = tar_act_m
        self.target_critic = tar_cri_m
        self.gamma = gamma
        self.tau = tau
        self.critic_optimizer = cri_optimizer
        self.actor_optimizer = act_optimizer

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_size))
        

    # Takes (s,a,r,s') obervation tuple as input
    def remember(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    #@tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.

        state_action_batch = [None for _ in range(self.batch_size)]

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)#, replace=False)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
