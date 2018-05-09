from keras import layers, models, optimizers, regularizers, initializers
from keras.layers import LeakyReLU
from keras import backend as K

import tensorflow as tf
import numpy as np
import copy
import random
from collections import namedtuple, deque

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, alpha):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.alpha = alpha

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')


        # Add hidden layers

        net = layers.Dense(units=300, kernel_regularizer=regularizers.l2(1e-5))(states)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)

        net = layers.Dense(units=400, kernel_regularizer=regularizers.l2(1e-5))(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)

        net = layers.Dense(units=200, kernel_regularizer=regularizers.l2(1e-5))(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)


        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size,
                                   activation='sigmoid',
                                   name='raw_actions')(net)     # if use tanh, rescale (output * 0.5) + 0.5

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
        #print('model.Actor.build_model -> actions', actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        print('setting policy update rate:', self.alpha)
        optimizer = optimizers.Adam(lr=self.alpha)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, beta):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.beta = beta

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=300, kernel_regularizer=regularizers.l2(1e-5))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)

        net_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(1e-5))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=300, kernel_regularizer=regularizers.l2(1e-5))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(1e-2)(net_actions)

        net_actions = layers.Dense(units=400, kernel_regularizer=regularizers.l2(1e-5))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(1e-2)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.add([net_states, net_actions])

        # Add more layers to the combined network if needed
        net = layers.Dense(units=200, kernel_regularizer=regularizers.l2(1e-5))(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1,
                                activation=None,
                                kernel_regularizer=regularizers.l2(1e-5),
                                kernel_initializer=initializers.RandomUniform(minval=-5e-3, maxval=5e-3),
                                # bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        print('setting value update rate:', self.beta)
        optimizer = optimizers.Adam(lr=self.beta)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)