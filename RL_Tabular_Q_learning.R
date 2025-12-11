
#Tabular Q-learning 
set.seed(42)

# Parameters for Price Process Simulation
# --- 0. Setup and Parameters (Réutilisés du code précédent) ---
set.seed(42)
N_hist <- 1000
S_hist <- cumsum(c(100, rnorm(N_hist, mean = 0, sd = 0.1))) 
S_initial <- 100 

Delta_S <- diff(S_hist)
sigma_hat <- sqrt(sum((Delta_S - mean(Delta_S))^2) / (length(Delta_S) - 1))

T_horizon <- 20
x_max <- 30
a_max <- 10
c_cost <- 0.02
lambda_pen <- 0.5
GAMMA <- 1.0
A_SIZE <- a_max + 1
ALPHA <- 0.1
EPISODES <- 500000 
N_S_BINS <- 31 
S_RANGE <- 3 * sigma_hat * sqrt(T_horizon)
S_min_bound <- S_initial - S_RANGE
S_max_bound <- S_initial + S_RANGE
S_grid <- seq(S_min_bound, S_max_bound, length.out = N_S_BINS + 1)

get_S_bin_index <- function(S_val) {
  s_bin <- cut(S_val, 
               breaks = S_grid, 
               labels = FALSE, 
               include.lowest = TRUE,
               right = FALSE) 
  s_bin <- pmin(pmax(s_bin, 1), N_S_BINS)
  return(s_bin)
}

Q_table <- array(0, dim = c(T_horizon, x_max + 1, N_S_BINS, A_SIZE))

step_env <- function(t, x, S, action_index,
                     T_horizon, x_max, sigma_hat, c_cost, lambda_pen) {
  # [Fonction step_env omise pour la concision, elle est supposée intacte]
  a <- action_index - 1
  a <- min(a, x)
  R_t <- a * S - c_cost * a^2
  x_next <- x - a
  z <- rnorm(1, mean = 0, sd = 1)
  S_next <- S + sigma_hat * z
  t_next <- t + 1
  done <- (t_next >= T_horizon) || (x_next <= 0)
  if (t_next >= T_horizon && x_next > 0) {
    liquidation_gain <- x_next * S_next - c_cost * x_next^2
    penalty <- - lambda_pen * x_next^2
    R_t <- R_t + liquidation_gain + penalty
    x_next <- 0
    done <- TRUE
  }
  list(t = t_next, x = x_next, S = S_next, reward = R_t, action_taken = a, done = done)
}

# --- 1. Boucle d'entraînement Q-Learning (doit être exécutée) ---

eps <- 1.0
eps_min <- 0.01
eps_decay <- 0.99999

cat("\nStarting Tabular Q-Learning (discretized)...\n")

for (ep in 1:EPISODES) {
  
  # Initialisation aléatoire de l'état (essentiel pour la couverture de l'espace)
  t <- 0
  x <- sample(0:x_max, 1)
  S <- runif(1, S_min_bound, S_max_bound) 
  
  total_reward <- 0
  
  while (t < T_horizon && x > 0) {
    
    # 1. État actuel (discretisé)
    t_idx <- t + 1 # 1-based [1..20]
    x_idx <- x + 1 # 1-based [1..31]
    s_idx <- get_S_bin_index(S) # 1-based [1..31]
    
    # 2. Choix de l'action (epsilon-greedy)
    feasible_a_max <- min(a_max, x)
    
    if (runif(1) < eps) {
      # Exploration: action aléatoire dans l'ensemble des actions faisables
      action_index <- sample(1:(feasible_a_max + 1), 1)
    } else {
      # Exploitation: argmax sur les Q-values stockées
      Qvals_slice <- Q_table[t_idx, x_idx, s_idx, ]
      
      # Masquer les actions infaisables avec une valeur très faible
      if (feasible_a_max < a_max) {
        # Action a=feasible_a_max+1 est l'index R 1-based: feasible_a_max+2
        infeasible_from <- feasible_a_max + 2 
        Qvals_slice[infeasible_from:A_SIZE] <- -1e9
      }
      
      action_index <- which.max(Qvals_slice)
    }
    
    # a est l'action choisie (0-based)
    a <- action_index - 1
    
    # 3. Transition vers l'état suivant
    step_result <- step_env(
      t, x, S, action_index,
      T_horizon, x_max, sigma_hat, c_cost, lambda_pen
    )
    
    # 4. État suivant (discretisé)
    t_next <- step_result$t
    x_next <- step_result$x
    S_next <- step_result$S
    reward <- step_result$reward
    done <- step_result$done
    
    # 1-based indices for next state
    t_next_idx <- t_next + 1
    x_next_idx <- x_next + 1
    s_next_idx <- get_S_bin_index(S_next)
    
    # 5. Calcul de la Cible de Bellman (Target)
    if (done) {
      target <- reward
    } else {
      # Q-value du max action dans l'état suivant
      # Utilisation d'indices R 1-based pour l'état suivant
      Q_next_slice <- Q_table[t_next_idx, x_next_idx, s_next_idx, ]
      
      # Ré-appliquer le masque d'inventaire pour l'état S' (si nécessaire, bien que théoriquement x' <= x_max)
      feasible_a_max_next <- min(a_max, x_next)
      if (feasible_a_max_next < a_max) {
        infeasible_from_next <- feasible_a_max_next + 2
        Q_next_slice[infeasible_from_next:A_SIZE] <- -1e9
      }
      
      target <- reward + GAMMA * max(Q_next_slice)
    }
    
    # 6. Mise à jour de la Q-Table (Q-Learning Update Rule)
    Q_current <- Q_table[t_idx, x_idx, s_idx, action_index]
    
    Q_table[t_idx, x_idx, s_idx, action_index] <- 
      Q_current + ALPHA * (target - Q_current)
    
    # Mise à jour des variables pour la prochaine itération
    t <- t_next
    x <- x_next
    S <- S_next
    total_reward <- total_reward + reward
    
    if (done) break
  }
  
  # Mise à jour de l'exploration
  eps <- max(eps_min, eps * eps_decay)
  
  if (ep %% 10000 == 0) {
    cat(sprintf("Episode %d | Total reward: %.2f | eps: %.6f\n",
                ep, total_reward, eps))
  }
}

cat("\nTabular Q-Learning finished.\n")

# --- 5. Visualisation de la Politique (Heatmap) ---

library(plotly)

# Choisissez un instant t fixe (par exemple, mi-parcours)
t_fixed <- 1 
t_idx_fixed <- t_fixed + 1 # 1-based index

x_grid <- 0:x_max
S_grid_viz <- S_grid[1:N_S_BINS] 

# Matrices pour l'Action (A) et la Valeur (V)
A_mat <- matrix(NA, nrow = length(x_grid), ncol = N_S_BINS)
V_mat <- matrix(NA, nrow = length(x_grid), ncol = N_S_BINS)


for (ix in seq_along(x_grid)) {
  x_val <- x_grid[ix]
  x_idx <- ix # 1-based index (0 -> 1, 30 -> 31)
  
  for (is in 1:N_S_BINS) {
    s_idx <- is
    
    if (x_val == 0) {
      # CORRECTION 1: Inventaire nul -> action 0, valeur 0 (état absorbant)
      A_mat[ix, is] <- 0
      V_mat[ix, is] <- 0 # L'inventaire est nul, donc la valeur future est nulle
    } else {
      # Tranche des Q-values
      Qvals_slice <- Q_table[t_idx_fixed, x_idx, s_idx, ]
      
      # Masquer les actions infaisables
      feasible_a_max <- min(a_max, x_val)
      
      # CORRECTION 2: Vérification stricte pour laquelle action est infaisable
      # L'action a=0 est l'indice R 1.
      # L'action a=feasible_a_max est l'indice R feasible_a_max + 1.
      infeasible_from <- feasible_a_max + 2 # L'indice R de la première action infaisable (a=feasible_a_max+1)
      
      if (infeasible_from <= A_SIZE) {
        Qvals_slice[infeasible_from:A_SIZE] <- -1e10 # Valeur très basse pour masquer
      }
      
      # CORRECTION 3: S'assurer que which.max ne soit pas appelé sur un vecteur vide,
      # bien que Qvals_slice devrait toujours être de longueur A_SIZE.
      # Le problème initial était que pour x=0, which.max était potentiellement appelé sur un vecteur ne contenant que -1e9,
      # ou sur une tranche non remplie de 0. La correction 1 gère x=0.
      
      # Calcul de l'action et de la valeur
      A_mat[ix, is] <- which.max(Qvals_slice) - 1 # Action optimale (0-based)
      V_mat[ix, is] <- max(Qvals_slice)          # Valeur optimale
    }
  }
}

# Inversion de l'axe Y pour la visualisation (Inventaire croissant vers le haut)
A_mat_flipped <- A_mat[rev(1:nrow(A_mat)), ]
V_mat_flipped <- V_mat[rev(1:nrow(V_mat)), ]
x_grid_flipped <- rev(x_grid)

# --- Tracé 1: Optimal Action Heatmap (Politique) ---

plot_q_learning_action_2d <- plot_ly(
  x = S_grid_viz, 
  y = x_grid_flipped, 
  z = A_mat_flipped, 
  type = "heatmap",
  colorscale = "Viridis",
  colorbar = list(title = "Optimal Action (a*)"),
  hovertemplate = "Price Bin Start (S): %{x:.2f}<br>Inventory (x): %{y}<br>Action (a*): %{z}<extra></extra>"
) %>%
  layout(
    title = list(text = sprintf("Tabular Q-Learning Optimal Action a*(x, S_bin) at t=%d/%d", 
                                t_fixed, T_horizon)),
    xaxis = list(title = "Asset Price (S) [Start of Bin]"),
    yaxis = list(title = "Inventory (x)", tickvals = seq(min(x_grid), max(x_grid), by = 5))
  )

print(plot_q_learning_action_2d)

# --- Tracé 2: Value Function Heatmap (Valeur) ---

plot_q_learning_value_2d <- plot_ly(
  x = S_grid_viz, 
  y = x_grid_flipped, 
  z = V_mat_flipped, 
  type = "heatmap",
  colorscale = "Plasma", # Une autre palette pour distinguer
  colorbar = list(title = "Value Function (V)"),
  hovertemplate = "Price Bin Start (S): %{x:.2f}<br>Inventory (x): %{y}<br>Value (V): %{z:.2f}<extra></extra>"
) %>%
  layout(
    title = list(text = sprintf("Tabular Q-Learning Value Function V(x, S_bin) at t=%d/%d", 
                                t_fixed, T_horizon)),
    xaxis = list(title = "Asset Price (S) [Start of Bin]"),
    yaxis = list(title = "Inventory (x)", tickvals = seq(min(x_grid), max(x_grid), by = 5))
  )

print(plot_q_learning_value_2d)

##Option 2 for visualization 

t_fixed <- 15 
t_idx_fixed <- t_fixed + 1 # 1-based index

x_grid <- 0:x_max

# Matrices pour l'Action (A) et la Valeur (V)
A_mat <- matrix(NA, nrow = length(x_grid), ncol = N_S_BINS)
V_mat <- matrix(NA, nrow = length(x_grid), ncol = N_S_BINS)


# --- 2. Boucle de Traçage Corrigée ---

for (ix in seq_along(x_grid)) {
  x_val <- x_grid[ix]
  x_idx <- ix # 1-based index (0 -> 1, 30 -> 31)
  
  for (is in 1:N_S_BINS) {
    s_idx <- is
    
    if (x_val == 0) {
      # Cas 1: Inventaire nul -> action 0, valeur 0 (état absorbant)
      A_mat[ix, is] <- 0
      V_mat[ix, is] <- 0 
    } else {
      # Cas 2: Inventaire > 0
      
      # Tranche des Q-values (copie pour modification)
      Qvals_slice <- Q_table[t_idx_fixed, x_idx, s_idx, ]
      
      # CORRECTION CRUCIALE: Remplacer les NA par 0 pour éviter le which.max(NA) -> length 0
      # Cela garantit que toutes les actions ont une Q-value numérique.
      Qvals_slice[is.na(Qvals_slice)] <- 0
      
      # Masquer les actions infaisables
      feasible_a_max <- min(a_max, x_val)
      
      # Indice R de la première action infaisable (a=feasible_a_max+1)
      infeasible_from <- feasible_a_max + 2 
      
      if (infeasible_from <= A_SIZE) {
        # Masquage par une valeur très basse
        Qvals_slice[infeasible_from:A_SIZE] <- -1e10 
      }
      
      # Calcul de l'action et de la valeur
      # which.max() va maintenant fonctionner car le vecteur est garanti sans NA
      A_mat[ix, is] <- which.max(Qvals_slice) - 1 # Action optimale (0-based)
      V_mat[ix, is] <- max(Qvals_slice)          # Valeur optimale
    }
  }
}

# Inversion de l'axe Y pour la visualisation (Inventaire croissant vers le haut)
A_mat_flipped <- A_mat[rev(1:nrow(A_mat)), ]
V_mat_flipped <- V_mat[rev(1:nrow(V_mat)), ]
x_grid_flipped <- rev(x_grid)

# --- 3. Tracé 1: Optimal Action Heatmap (Politique) ---

plot_q_learning_action_2d <- plot_ly(
  x = S_grid_viz, 
  y = x_grid_flipped, 
  z = A_mat_flipped, 
  type = "heatmap",
  colorbar = list(title = "Optimal Action (a*)"),
  hovertemplate = "Price Bin Start (S): %{x:.2f}<br>Inventory (x): %{y}<br>Action (a*): %{z}<extra></extra>"
) %>%
  layout(
    title = list(text = sprintf("Tabular Q-Learning Optimal Action a*(x, S_bin) at t=%d/%d (Politique)", 
                                20-t_fixed, T_horizon)),
    xaxis = list(title = "Asset Price (S) [Start of Bin]"),
    yaxis = list(title = "Inventory (x)", tickvals = seq(min(x_grid), max(x_grid), by = 5))
  )

print(plot_q_learning_action_2d)

# --- 4. Tracé 2: Value Function Heatmap (Valeur) ---

plot_q_learning_value_2d <- plot_ly(
  x = S_grid_viz, 
  y = x_grid_flipped, 
  z = V_mat_flipped, 
  type = "heatmap",
  colorscale = "Plasma", 
  colorbar = list(title = "Value Function (V)"),
  hovertemplate = "Price Bin Start (S): %{x:.2f}<br>Inventory (x): %{y}<br>Value (V): %{z:.2f}<extra></extra>"
) %>%
  layout(
    title = list(text = sprintf("Tabular Q-Learning Value Function V(x, S_bin) at t=%d/%d (Valeur)", 
                                20-t_fixed, T_horizon)),
    xaxis = list(title = "Asset Price (S) [Start of Bin]"),
    yaxis = list(title = "Inventory (x)", tickvals = seq(min(x_grid), max(x_grid), by = 5))
  )

print(plot_q_learning_value_2d)
