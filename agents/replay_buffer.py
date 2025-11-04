import numpy as np

class ReplayBuffer:
    """
    Simple replay buffer for off-policy algorithms like MADQN.
    It stores (state, action, reward, next_state, done) tuples
    and samples random minibatches for training.
    """

    def __init__(self, state_dim, action_dim, capacity=int(1e5)):
        """
        Initializes the replay buffer.

        Args:
            state_dim (int or tuple): Dimension of the state space (flattened automatically if tuple)
            action_dim (int): Dimension of the action space
            capacity (int): Maximum number of transitions to store
        """
        # Auto-handle tuple inputs like (5,5)
        if isinstance(state_dim, tuple):
            self.state_dim = int(np.prod(state_dim))
        else:
            self.state_dim = int(state_dim)

        self.action_dim = action_dim
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Preallocate memory for efficiency
        self.states = np.zeros((capacity, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, s2, done):
        """
        Stores one transition in the buffer.
        Automatically flattens observations if needed.
        """
        s = np.array(s).flatten()
        s2 = np.array(s2).flatten()

        # Sanity check: match observation shape to buffer
        if s.shape[0] != self.state_dim:
            print(f"[WARN] State shape mismatch: got {s.shape}, expected {self.state_dim}")
        if s2.shape[0] != self.state_dim:
            print(f"[WARN] Next-state shape mismatch: got {s2.shape}, expected {self.state_dim}")

        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr] = done

        # Circular pointer update
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=64):
        """
        Returns a random batch of transitions for training.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs]
        )
        return batch

    def __len__(self):
        return self.size
