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
LR_G = 1e-4
LR_D = 3e-4           # Higher critic LR for more stable WGAN
LR_E = 1e-4
N_CRITIC = 3           # Fewer critic updates, more G updates
LAMBDA_GP = 10.0       # Gradient penalty weight
LAMBDA_COL = 20.0      # Collision loss weight (increased)
LAMBDA_LEN = 0.5       # Path length weight
LAMBDA_SMOOTH = 1.0    # Smoothness weight
LAMBDA_RECON = 2.0     # Reconstruction weight (E_path, increased)
LAMBDA_CONVEXITY = 0.0 # Convexity off for now (add back later)
NUM_EPOCHS = 2000      # Much more training

# --- Online Inference ---
LAMBDA_CONTINUITY = 0.1   # Latent continuity weight (lower = more flexible)
ONLINE_LR = 0.05          # Online gradient descent step size
ONLINE_STEPS = 20         # Gradient descent steps per replanning cycle
REPLAN_HZ = 20            # Replanning frequency (Hz)
