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
LR_D = 1e-4
LR_E = 1e-4
N_CRITIC = 5            # D updates per G/E update
LAMBDA_GP = 10.0        # Gradient penalty weight
LAMBDA_COL = 10.0       # Collision loss weight
LAMBDA_LEN = 0.5        # Path length weight
LAMBDA_SMOOTH = 1.0     # Smoothness weight
LAMBDA_RECON = 1.0      # Reconstruction weight (E_path)
LAMBDA_CONVEXITY = 2.0  # Convexity regularization weight
NUM_EPOCHS = 500        # Phase 1 training epochs

# --- Online Inference ---
LAMBDA_CONTINUITY = 0.5  # Latent continuity weight
ONLINE_LR = 0.01         # Online gradient descent step size
ONLINE_STEPS = 5         # Gradient descent steps per replanning cycle
REPLAN_HZ = 10           # Replanning frequency (Hz)
