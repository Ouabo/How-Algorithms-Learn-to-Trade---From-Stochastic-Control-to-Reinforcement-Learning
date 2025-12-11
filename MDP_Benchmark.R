# --- MDP / Dynamic Programming Implementation --- The distribution of the next reward is N(S_t, sigma_hat^2)



library(plotly) # package for visualization

set.seed(42)
N_hist <- 1000
# Random walk series
S_hist <- cumsum(c(100, rnorm(N_hist, mean = 0, sd = 0.1))) 
S_initial <- 100 

# Calibration of sigma
Delta_S <- diff(S_hist)
sigma_hat <- sqrt(sum((Delta_S - mean(Delta_S))^2) / (length(Delta_S) - 1))

# model Param
T_horizon <- 20
x_max <- 30
a_max <- 10
c_cost <- 0.02
lambda_pen <- 0.5
GAMMA <- 1.0 # Undiscounted
A_SIZE <- a_max + 1

# Discretization of the price space S
N_S_BINS <- 31
S_RANGE <- 3 * sigma_hat * sqrt(T_horizon)
S_min_bound <- S_initial - S_RANGE
S_max_bound <- S_initial + S_RANGE
S_grid <- seq(S_min_bound, S_max_bound, length.out = N_S_BINS + 1)
S_midpoints <- (S_grid[-1] + S_grid[-(N_S_BINS + 1)]) / 2

cat("Calibrated Sigma :", sigma_hat, "\n")

# Calculate Transition Probabilities P(s' | s) ---

# P_transition[s, s'] is the probability of transitioning from bin s to bin s'.
P_transition <- matrix(0, nrow = N_S_BINS, ncol = N_S_BINS)

cat("Calculating Transition Matrix P(s' | s)...\n")

for (s_idx in 1:N_S_BINS) {
  # The price S_t is approximated by the midpoint of bin s.
  S_t <- S_midpoints[s_idx]
  
  # The distribution of the following price is N(S_t, sigma_hat^2)
  for (s_prime_idx in 1:N_S_BINS) {
    S_lower <- S_grid[s_prime_idx]
    S_upper <- S_grid[s_prime_idx + 1]
    
    # P(S_lower < S_{t+1} < S_upper)
    prob <- pnorm(S_upper, mean = S_t, sd = sigma_hat) - 
      pnorm(S_lower, mean = S_t, sd = sigma_hat)
    
    P_transition[s_idx, s_prime_idx] <- prob
  }
  
  # Normalization (ensures sum(P)=1 despite bin clipping)
  P_transition[s_idx, ] <- P_transition[s_idx, ] / sum(P_transition[s_idx, ])
}


#Dynamic Programming   ---

V_table <- array(0, dim = c(T_horizon, x_max + 1, N_S_BINS)) 
A_table <- array(NA, dim = c(T_horizon, x_max + 1, N_S_BINS))

cat("Starting Dynamic Programming (Backward Induction)...\n")

# t ranges from T-1 (last decision step) to 0 (first step)
for (t in (T_horizon - 1):0) {
  t_idx <- t + 1 
  
  for (x in 0:x_max) {
    x_idx <- x + 1
    
    for (s in 1:N_S_BINS) {
      S_t <- S_midpoints[s]
      
      max_val <- -Inf
      optimal_a <- NA
      
      feasible_a_max <- min(a_max, x)
      
      
      if (x == 0) {
        V_table[t_idx, x_idx, s] <- 0
        A_table[t_idx, x_idx, s] <- 0
        next
      }
      
      
      for (a in 0:feasible_a_max) {
        
        R_t <- a * S_t - c_cost * a^2 
        x_next <- x - a
        E_V_next <- 0
        
        
        if (t + 1 == T_horizon) {
          
          # Calculation of the final liquidation expectation and penalty
          expected_terminal_gain <- 0
          for (s_prime in 1:N_S_BINS) {
            S_next <- S_midpoints[s_prime]
            
            liquidation_gain <- x_next * S_next - c_cost * x_next^2
            penalty <- - lambda_pen * x_next^2
            
            prob_s_prime <- P_transition[s, s_prime]
            
            expected_terminal_gain <- expected_terminal_gain + 
              prob_s_prime * (liquidation_gain + penalty)
          }
          E_V_next <- expected_terminal_gain
          
        } else if (x_next == 0) {
          # Condition 2: Inventory liquidated before T
          E_V_next <- 0
          
        } else {
          # Condition 3: Standard transition: E[V_{t+1}(x', s')]
          x_next_idx <- x_next + 1
          V_next_slice <- V_table[t_idx + 1, x_next_idx, ] 
          
          E_V_next <- sum(P_transition[s, ] * V_next_slice)
        }
        
        # Bellman Equation for Q-Value (Action-Value)
        current_Q_val <- R_t + GAMMA * E_V_next
        
        # Maximisation
        if (current_Q_val > max_val) {
          max_val <- current_Q_val
          optimal_a <- a 
        }
      }
      
      
      V_table[t_idx, x_idx, s] <- max_val
      A_table[t_idx, x_idx, s] <- optimal_a
    }
  }
  
  if (t %% 5 == 0) {
    cat(sprintf("DP Iteration: t = %d completed.\n", t))
  }
}

cat("\nDynamic Programming finished.\n")

#  Visualization 

t_fixed <- 0 
t_idx_fixed <- t_fixed + 1

x_grid <- 0:x_max
S_grid_viz <- S_midpoints # Midpoints 

# Extraction des résultats
A_mat <- A_table[t_idx_fixed, , ] 
V_mat <- V_table[t_idx_fixed, , ] 

# Reversal of the Y axis for easier reading (Inventory increasing upwards)
A_mat_flipped <- A_mat[rev(1:nrow(A_mat)), ]
V_mat_flipped <- V_mat[rev(1:nrow(V_mat)), ]
x_grid_flipped <- rev(x_grid)

# ---  Optimal Action Heatmap  ---

plot_mdp_action_2d <- plot_ly(
  x = S_grid_viz, 
  y = x_grid_flipped, 
  z = A_mat_flipped, 
  type = "heatmap",
  colorscale = "Viridis",
  colorbar = list(title = "Optimal Action (a*)"),
  hovertemplate = "Price S: %{x:.2f}<br>Inventory x: %{y}<br>Action a*: %{z}<extra></extra>"
) %>%
  layout(
    title = list(text = sprintf("MDP Optimal Control Policy a*(x, S_bin) at t=%d/%d", 
                                t_fixed, T_horizon)),
    xaxis = list(title = "Asset Price (S) [Midpoint of Bin]"),
    yaxis = list(title = "Inventory (x)", tickvals = seq(min(x_grid), max(x_grid), by = 5))
  )

print(plot_mdp_action_2d)

# ---  Value Function Heatmap  ---

plot_mdp_value_2d <- plot_ly(
  x = S_grid_viz, 
  y = x_grid_flipped, 
  z = V_mat_flipped, 
  type = "heatmap",
  colorscale = "Plasma", 
  colorbar = list(title = "Value Function (V)"),
  hovertemplate = "Price S: %{x:.2f}<br>Inventory x: %{y}<br>Value V: %{z:.2f}<extra></extra>"
) %>%
  layout(
    title = list(text = sprintf("MDP Value Function V(x, S_bin) at t=%d/%d", 
                                t_fixed, T_horizon)),
    xaxis = list(title = "Asset Price (S) [Midpoint of Bin]"),
    yaxis = list(title = "Inventory (x)", tickvals = seq(min(x_grid), max(x_grid), by = 5))
  )

print(plot_mdp_value_2d)