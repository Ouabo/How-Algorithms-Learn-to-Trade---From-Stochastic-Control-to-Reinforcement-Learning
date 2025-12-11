# Load necessary library for visualization
# Ensure you have the 'plotly' library installed: install.packages("plotly")
# --- MDP / Dynamic Programming Implementation --- The distribution of the next price is approximated using Monte Carlo simulation.

library(plotly)

set.seed(42)

##############################
# 1. Load or define price data
##############################

# Example: you would replace this with real data:
# Exple: S_hist <- your vector of historical daily prices
# Here we create a dummy price series just for illustration:
N_hist <- 1000
#S_hist <- cumsum(c(100, rnorm(N_hist, mean = 0, sd = 0.1)))  # random walk around 100
S_hist <- cumsum(c(100, rnorm(N_hist, mean = 0, sd = 0.1)))
##########################################
# 2. Calibration of sigma from price data
##########################################

# We use the standard deviation of price increments (Delta_S) as the volatility parameter (sigma_hat).
# The original code's calculation of sigma_hat already accounts for the sample mean of increments
# being near zero, which is appropriate for a model where the drift is ignored (or assumed zero).
Delta_S <- diff(S_hist)
Delta_S_bar <- mean(Delta_S) # This is the mean, but we assume mu_hat = 0 for the forward process.
sigma_hat <- sqrt(sum((Delta_S - Delta_S_bar)^2) / (length(Delta_S) - 1))

cat("Calibrated sigma (standard deviation of increments, assuming mu=0):", sigma_hat, "\n")
# Forward price evolution will be: S_next = S_val + sigma_hat * z, where z ~ N(0,1)

###################################################
# 3. Model parameters and discretization of states
###################################################

T_horizon <- 20    # number of decision steps (t = 0,...,T-1)
x_max     <- 30    # maximum inventory
a_max     <- 10    # maximum action per step
c_cost    <- 0.02  # trading cost coefficient (e.g., c * a^2)
M_mc      <- 2000  # number of Monte Carlo samples for expected value calculation
lambda_pen <- 0.5 # strength of leftover inventory penalty (tune as you like)

# Inventory grid: x_i in {0,1,...,x_max}
x_grid <- 0:x_max
I <- length(x_grid)

# Price grid: choose min/max around historical range
S_min <- S_min_bound # quantile(S_hist, 0.05)
S_max <- S_max_bound #quantile(S_hist, 0.95)
J     <- 21               # number of price grid points
S_grid <- seq(S_min, S_max, length.out = J)
S_grid <- seq(S_min_bound, S_max_bound, length.out = J)

cat("Price grid from", S_min, "to", S_max, "with", J, "points.\n")

###########################################
# 4. Containers for V_t and optimal policy
###########################################

# We store V_t for t = 0,...,T.
# V_list[[t+1]] is the matrix of V_t, dimension I (inventory) x J (price)
V_list <- vector("list", T_horizon + 1)

# Also store optimal action a_t^* at each grid point (same dimension)
A_list <- vector("list", T_horizon)

###########################################################
# 5. Terminal condition: V_T(x,S) (Forced liquidation + penalty)
###########################################################

V_T <- matrix(0, nrow = I, ncol = J)

for (i in 1:I) {
  x_val <- x_grid[i]
  # Calculate index of x_val within x_grid (x_grid starts at 0, so index i corresponds to x=i-1, 
  # but here i is used in 1:I, so x_grid[i] is the value)
  
  for (j in 1:J) {
    S_val <- S_grid[j]
    
    # Forced liquidation at final time (x * S - cost * x^2)
    liquidation_gain <- x_val * S_val - c_cost * x_val^2
    
    # Additional quadratic penalty for leftover inventory
    penalty <- - lambda_pen * x_val^2
    
    # Combined terminal value
    V_T[i, j] <- liquidation_gain + penalty
  }
}

V_list[[T_horizon + 1]] <- V_T

#####################################
# 6. Helper: linear interpolation in S
#####################################

# Given V_next (I x J) = V_{t+1}(x_i, S^{(j)}),
# x_index in 1:I, and a price S_val,
# return an approximate V_{t+1}(x_i, S_val) by linear interpolation.
interp_V_next <- function(V_next, x_index, S_val, S_grid) {
  # If S_val is outside the grid, clamp to edge (Extrapolation is avoided)
  n_S <- length(S_grid)
  if (S_val <= S_grid[1]) {
    return(V_next[x_index, 1])
  }
  if (S_val >= S_grid[n_S]) {
    return(V_next[x_index, n_S])
  }
  
  # Find j such that S_grid[j] <= S_val <= S_grid[j+1]
  # This uses max(which(...)) which is correct for finding the lower bound index
  j <- max(which(S_grid <= S_val))
  
  # If S_val is exactly the last point (which should be caught by the clamping check, but for safety)
  if (j == n_S) {
    return(V_next[x_index, j])
  }
  
  S_low <- S_grid[j]
  S_high <- S_grid[j + 1]
  
  # Linear interpolation weight (lambda = weight for S_low)
  lambda <- (S_high - S_val) / (S_high - S_low)
  
  # Interpolation calculation
  V_low  <- V_next[x_index, j]
  V_high <- V_next[x_index, j + 1]
  V_interp <- lambda * V_low + (1 - lambda) * V_high
  
  return(V_interp)
}

############################################################
# 7. Backward induction: compute V_t and optimal policy A_t
############################################################

# [Image of a dynamic programming grid showing a state (x, S) at time t, 
# with arrows pointing to the next time step t+1 and showing the optimization process]

for (t in (T_horizon-1):0) {
  cat("Computing V_", t, "...\n", sep = "")
  
  V_next <- V_list[[t + 2]]   # V_{t+1}, matrix I x J
  V_t    <- matrix(0, nrow = I, ncol = J)
  A_t    <- matrix(0, nrow = I, ncol = J)  # optimal actions
  
  # Pre-simulate shocks for this entire time step calculation
  z_vec <- rnorm(M_mc, mean = 0, sd = 1)
  
  # Loop over all grid states (x_i, S^{(j)})
  for (i in 1:I) {
    x_val <- x_grid[i]
    
    # Action set A(x_i): 0 up to min(a_max, x_val)
    a_max_i <- min(a_max, x_val)
    actions <- 0:a_max_i
    
    # Get the indices of the next inventory states (x_next = x_val - a).
    # Since x_grid is 0:x_max, x_grid[i] = i-1.
    # x_next = x_val - a = (i-1) - a.
    # x_next_index = x_next + 1 = i - a.
    # We use these indices to look up V_next (x-dimension is row index)
    x_next_indices <- i - actions 
    
    # If no inventory, the only action is 0. V_t(0, S) is 0 because V_T(0, S) = 0.
    if (x_val == 0) {
      V_t[i, ] <- 0
      A_t[i, ] <- 0
      next
    }
    
    for (j in 1:J) {
      S_val <- S_grid[j]
      
      # Compute Q-values for all available actions 'a'
      Q_values <- numeric(length(actions))
      
      for (k in seq_along(actions)) {
        a <- actions[k]
        
        # Immediate reward: r_t(x, S, a) = a * S - c_cost * a^2
        immediate <- a * S_val - c_cost * a^2
        
        # Future expected value: Monte Carlo estimate E[V_{t+1}(x_{next}, S_{next})]
        
        # 1. Calculate future prices (Additive Random Walk model: S_next = S_val + sigma_hat * z)
        # No drift (mu=0) as required.
        S_next_vec <- S_val + sigma_hat * z_vec
        
        # 2. Get the index of the next inventory state
        x_next_index <- x_next_indices[k]
        
        # 3. Interpolate V_next for all S_next samples and take the mean
        # Using sapply for functional, slightly more readable iteration over MC samples
        V_future_sum <- sum(sapply(S_next_vec, function(S_next) {
          interp_V_next(V_next, x_next_index, S_next, S_grid)
        }))
        
        V_future_mc <- V_future_sum / M_mc
        
        Q_values[k] <- immediate + V_future_mc
      }
      
      # Optimal value and action (find action that maximizes Q-value)
      best_idx <- which.max(Q_values)
      V_t[i, j] <- Q_values[best_idx]
      A_t[i, j] <- actions[best_idx]
    }
  }
  
  V_list[[t + 1]] <- V_t
  A_list[[t + 1]] <- A_t
}

cat("Backward induction complete.\n")

############################################
#############Visualization##############################

# Helper function to extract optimal action for a specific (t, x, S) state
get_opt_action <- function(t, x, S, A_list, x_grid, S_grid) {
  # t: time index (0,...,T_horizon-1)
  # x: current inventory (must be in x_grid)
  # S: current price
  # returns optimal action a_t^*(x,S) using nearest price grid
  
  A_t <- A_list[[t + 1]]       # matrix I x J
  
  # Find row index for inventory x
  i <- which(x_grid == x)
  if (length(i) == 0) stop("x not on grid")
  
  # Find column index for price S (nearest grid point)
  j <- which.min(abs(S_grid - S))
  
  a_opt <- A_t[i, j]
  return(a_opt)
}

# --- Visualization Parameters ---
t_plot <- 19 # Choose which time step to visualize:

# Extract value function and optimal policy at time t_plot
V_t <- V_list[[t_plot + 1]]    # I x J matrix
A_t <- A_list[[t_plot + 1]]    # I x J matrix

# Sanity check dimensions:
cat("\nSanity check: V_t dimensions:", dim(V_t), "\n")
cat("Sanity check: A_t dimensions:", dim(A_t), "\n")

################################################
## 1) 3D Surface Plot: Value Function V_t(x, S)
################################################

p_V_3d <- plot_ly(
  x = ~S_grid,         # price (x-axis)
  y = ~x_grid,         # inventory (y-axis)
  z = ~V_t,            # matrix: rows = y, cols = x
  type = "surface"
) %>%
  layout(
    title = paste0("Value Function V_t(x, S) at t = ", t_plot),
    scene = list(
      xaxis = list(title = "Price S"),
      yaxis = list(title = "Inventory x"),
      zaxis = list(title = "V_t(x, S)")
    )
  )

#p_V_3d

######################################################
## 2) 3D Surface Plot: Optimal Control a_t^*(x, S)
######################################################

p_A_3d <- plot_ly(
  x = ~S_grid,
  y = ~x_grid,
  z = ~A_t,
  type = "surface"
) %>%
  layout(
    title = paste0("Optimal Action a*_t(x, S) at t = ", t_plot),
    scene = list(
      xaxis = list(title = "Price S"),
      yaxis = list(title = "Inventory x"),
      zaxis = list(title = "Optimal action a*")
    )
  )

#p_A_3d

#############################################
## 3) 2D Heatmap: Value Function V_t(x, S)
#############################################

p_V_2d <- plot_ly(
  x = ~S_grid,
  y = ~x_grid,
  z = ~V_t,
  type = "heatmap",
  colorscale = "Plasma",
) %>%
  layout(
    title = paste0(" MDP (Transi. Prob. Approx-MC) Value Function V_t(x, S) at t = ", t_plot, " (heatmap)"),
    xaxis = list(title = "Price S"),
    yaxis = list(title = "Inventory x")
  )

p_V_2d

#################################################
## 4) 2D Heatmap: Optimal Control a_t^*(x, S)
#################################################

p_A_2d <- plot_ly(
  x = ~S_grid,
  y = ~x_grid,
  z = ~A_t,
  type = "heatmap"
) %>%
  layout(
    title = paste0("MDP (Transi. Prob. Approx-MC) Optimal Action a*_t(x, S) at t = ", t_plot, " (heatmap)"),
    xaxis = list(title = "Price S"),
    yaxis = list(title = "Inventory x")
  )

p_A_2d