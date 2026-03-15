import gym  # type: ignore # pyre-ignore
from gym import spaces  # type: ignore # pyre-ignore
import numpy as np  # type: ignore # pyre-ignore
import pymunk  # type: ignore # pyre-ignore
import pygame  # type: ignore # pyre-ignore
import math
from typing import Any

class DoublePendulumEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, reward_type='shaped'):
        super().__init__()
        self.reward_type = reward_type
        self.dt = 1.0 / 60.0
        self.screen_width = 800
        self.screen_height = 600
        
        # Action space: Continuous force applied to the cart [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obs space: Cart x, Cart vx, Pole1 angle, Pole1 angular vel, Pole2 angle, Pole2 angular vel
        high = np.array([np.inf, np.inf, np.pi, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, shape=(6,), dtype=np.float32)
        
        self.screen: Any = None
        self.clock: Any = None
        self.space: Any = None

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, -981)

        # Track
        self.track = pymunk.Segment(self.space.static_body, (50, 300), (750, 300), 5)
        self.track.friction = 0.0

        # Cart
        mass_cart = 1.0
        size_cart = (50, 30)
        moment_cart = pymunk.moment_for_box(mass_cart, size_cart)
        self.cart_body = pymunk.Body(mass_cart, moment_cart)
        self.cart_body.position = (400, 300)
        self.cart_shape = pymunk.Poly.create_box(self.cart_body, size_cart)
        self.cart_shape.friction = 0.0
        
        # Groove joint to keep cart on track
        self.groove = pymunk.GrooveJoint(self.space.static_body, self.cart_body, (50, 300), (750, 300), (0, 0))

        # Pole 1
        mass_pole1 = 0.5
        length_pole1 = 100
        moment_pole1 = pymunk.moment_for_segment(mass_pole1, (0, 0), (0, length_pole1), 5) * 1.5
        self.pole1_body = pymunk.Body(mass_pole1, moment_pole1)
        self.pole1_body.position = (400, 300)
        self.pole1_shape = pymunk.Segment(self.pole1_body, (0, 0), (0, length_pole1), 5)
        
        # Pivot joint cart -> pole 1
        self.pivot1 = pymunk.PivotJoint(self.cart_body, self.pole1_body, (400, 300))
        self.pivot1.max_force = 50000
        self.pivot1.error_bias = 0.2

        # Pole 2
        mass_pole2 = 0.5
        length_pole2 = 100
        moment_pole2 = pymunk.moment_for_segment(mass_pole2, (0, 0), (0, length_pole2), 5) * 1.5
        self.pole2_body = pymunk.Body(mass_pole2, moment_pole2)
        self.pole2_body.position = (400, 400)
        self.pole2_shape = pymunk.Segment(self.pole2_body, (0, 0), (0, length_pole2), 5)
        
        # Pivot joint pole 1 -> pole 2
        self.pivot2 = pymunk.PivotJoint(self.pole1_body, self.pole2_body, (400, 400))
        self.pivot2.max_force = 50000
        self.pivot2.error_bias = 0.2

        self.space.add(self.track, self.cart_body, self.cart_shape, self.groove, 
                       self.pole1_body, self.pole1_shape, self.pivot1, 
                       self.pole2_body, self.pole2_shape, self.pivot2)

        return self._get_obs()

    def _get_obs(self):
        # Normalize positions relative to screen center
        cart_x = (self.cart_body.position.x - 400) / 400.0
        cart_vx = self.cart_body.velocity.x / 100.0
        
        # Pymunk angles need normalizing to [-pi, pi]
        theta1 = (self.pole1_body.angle + np.pi) % (2 * np.pi) - np.pi
        omega1 = self.pole1_body.angular_velocity
        
        theta2 = (self.pole2_body.angle + np.pi) % (2 * np.pi) - np.pi
        omega2 = self.pole2_body.angular_velocity
        
        return np.array([cart_x, cart_vx, theta1, omega1, theta2, omega2], dtype=np.float32)

    def step(self, action):
        force_mag = action[0] * 5000.0 # Scale action [-1, 1] to force
        self.cart_body.apply_force_at_local_point((force_mag, 0), (0, 0))
        
        # Physics Sub-Stepping for better stability
        physics_steps = 4
        for _ in range(physics_steps):
            self.space.step(self.dt / physics_steps)
            
        obs = self._get_obs()
        
        cart_x, _, theta1, omega1, theta2, omega2 = obs

        # Baseline Reward
        baseline_reward = math.cos(theta1) + math.cos(theta2)
        
        # Shaped Reward
        if self.reward_type == 'shaped':
            center_penalty = abs(cart_x) * 0.1
            vel_penalty = (abs(omega1) + abs(omega2)) * 0.01
            action_penalty = (action[0]**2) * 0.001
            reward = baseline_reward - center_penalty - vel_penalty - action_penalty
        else:
            reward = baseline_reward

        # Episode ends if cart goes off screen
        done = bool(abs(cart_x) > 1.0)

        return obs, reward, done, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        
        # Pymunk coordinates have y=0 at bottom, pygame has y=0 at top. Needs flipping.
        def flipy(y): return self.screen_height - int(y)

        # Draw track
        pygame.draw.line(self.screen, (0, 0, 0), (50, flipy(300)), (750, flipy(300)), 5)
        
        # Draw cart
        cx, cy = int(self.cart_body.position.x), flipy(self.cart_body.position.y)
        pygame.draw.rect(self.screen, (0, 0, 255), (cx-25, cy-15, 50, 30))

        # Draw poles
        p1_start = (int(self.pole1_body.position.x), flipy(self.pole1_body.position.y))
        p1_end = (int(p1_start[0] - 100 * math.sin(self.pole1_body.angle)), 
                  int(p1_start[1] - 100 * math.cos(self.pole1_body.angle)))
        pygame.draw.line(self.screen, (255, 0, 0), p1_start, p1_end, 5)

        p2_start = p1_end
        p2_end = (int(p2_start[0] - 100 * math.sin(self.pole2_body.angle)), 
                  int(p2_start[1] - 100 * math.cos(self.pole2_body.angle)))
        pygame.draw.line(self.screen, (0, 255, 0), p2_start, p2_end, 5)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
