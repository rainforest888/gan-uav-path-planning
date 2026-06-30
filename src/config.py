"""Global configuration constants for GAN-UAV path planning."""

# --- Environment ---
VOXEL_RES = 32          # Voxel grid resolution (VOXEL_RES^3 for 3D)
SPACE_SIZE = 10.0       # World size in meters (cube: SPACE_SIZE^dim)
OBSTACLE_MIN_R = 0.5    # Min obstacle radius
OBSTACLE_MAX_R = 1.5    # Max obstacle radius
NUM_OBSTACLES_MIN = 3   # Min obstacles per scene
NUM_OBSTACLES_MAX = 10  # Max obstacles per scene

# --- Path Representation ---
K_WAYPOINTS = 10        # Key waypoints output by generator
N_TRAJECTORY = 100      # Trajectory points after interpolation

# --- Generator ---
LATENT_DIM = 128        # z dimension
CONDITION_DIM = 256     # c dimension (from E_env)

# --- Training ---
BATCH_SIZE = 64
LR_G = 5e-5             # Lower G LR for stability
LR_D = 2e-4             # Moderate critic LR
LR_E = 5e-5             # Lower E LR
N_CRITIC = 5            # Standard WGAN-GP ratio
LAMBDA_GP = 10.0        # Gradient penalty weight
LAMBDA_COL = 5.0        # Collision loss weight (moderate)
LAMBDA_LEN = 0.1        # Path length weight (reduced)
LAMBDA_SMOOTH = 0.5     # Smoothness weight (reduced)
LAMBDA_RECON = 5.0      # Reconstruction weight (stronger - key for quality)
LAMBDA_CONVEXITY = 1.0  # Convexity regularization (moderate)
NUM_EPOCHS = 300        # Shorter: better to have clean early checkpoint

# --- Online Inference ---
LAMBDA_CONTINUITY = 0.05  # Latent continuity weight (very flexible)
ONLINE_LR = 0.1           # Larger step for faster convergence
ONLINE_STEPS = 30         # More optimization steps
REPLAN_HZ = 20            # Replanning frequency (Hz)
