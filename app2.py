import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time

# --- 1. PyTorch Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# --- 2. Data Loading & Model Loading Logic ---

@st.cache_data
def load_data(filepath):
    """Loads and processes the player data, returning a DataFrame of season averages."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The data file '{filepath}' was not found.")
        return None
    features = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%']
    player_pool = df.groupby('Player')[features].mean().reset_index()
    return player_pool

@st.cache_resource
def load_model(model_path, input_size):
    """Loads the pre-trained PyTorch model."""
    model = MLP(input_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error(f"Error: The model file '{model_path}' was not found. Please ensure it is in the correct directory.")
        return None
    model.eval()  # Set the model to evaluation mode
    return model

# NEW: Function to get the AI's score for a specific team
def get_ai_score(team_df, model, player_pool_df):
    """
    Predicts the team score using the fixed PyTorch ANN model.
    """
    if model is None:
        return np.nan

    features = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%']
    
    # 1. Scale the team stats using the full player pool's scale (needed for consistency)
    scaler = MinMaxScaler()
    scaler.fit(player_pool_df[features]) 
    
    # Get the sum of the team's unscaled stats
    input_vector = team_df[features].sum().values.reshape(1, -1)
    
    # Scale the sum *correctly* based on how the original data was scaled (0-1)
    # The ANN was trained on the sum of 5 scaled players. We must reproduce that.
    
    # Simpler approach: Scale the *individual* player stats and then sum them up.
    team_df_scaled = scaler.transform(team_df[features])
    input_vector_scaled = team_df_scaled.sum(axis=0)

    # 2. Run Inference
    with torch.no_grad():
        input_tensor = torch.tensor(input_vector_scaled, dtype=torch.float32).view(1, -1)
        predicted_score = model(input_tensor).item()
        
    return predicted_score

# --- 3. Optimization and Simulation Logic (Unchanged) ---

def find_optimal_team_custom(player_pool, weights, label="Team"):
    """Searches for the optimal team based on the provided weights."""
    if player_pool is None:
        return None, None, 0

    features = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%']
    scaler = MinMaxScaler()
    player_pool_scaled = player_pool.copy()
    player_pool_scaled[features] = scaler.fit_transform(player_pool_scaled[features])

    best_team_indices = None
    best_score = -np.inf
    
    all_stats_scaled = player_pool_scaled[features].values
    player_ids = player_pool.index.to_numpy()
    weight_vector = np.array([
        weights['PTS'], weights['AST'], weights['TRB'], 
        weights['STL'], weights['BLK'], weights['FG%']
    ])

    search_space_size = 30000 
    
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text=f"Simulating {label} lineups...")
    start_time = time.time()

    for i in range(search_space_size):
        team_indices = np.random.choice(player_ids, 5, replace=False)
        team_stats_matrix = all_stats_scaled[team_indices]
        team_total_stats = np.sum(team_stats_matrix, axis=0)
        calculated_score = np.dot(team_total_stats, weight_vector)

        if calculated_score > best_score:
            best_score = calculated_score
            best_team_indices = team_indices
        
        if (i + 1) % 5000 == 0:
            progress_bar.progress(int((i + 1) / search_space_size * 100), text=f"Simulating {label}: Tested {i+1}/{search_space_size}")

    end_time = time.time()
    progress_bar.empty()
    progress_placeholder.empty()

    optimal_team_df = player_pool.loc[best_team_indices]
    return optimal_team_df, best_score, end_time - start_time


def run_single_game(team_alpha, team_beta, score_alpha, score_beta, possessions=105, base_ppp=1.08):
    """Simulates a single game based on optimized scores AND actual defensive stats."""
    
    def_alpha = team_alpha['STL'].sum() + team_alpha['BLK'].sum()
    def_beta = team_beta['STL'].sum() + team_beta['BLK'].sum()
    
    total_score_strength = score_alpha + score_beta
    share_alpha = score_alpha / total_score_strength
    expected_total_points = possessions * 2 * base_ppp

    expected_pts_alpha = expected_total_points * share_alpha
    expected_pts_beta = expected_total_points * (1 - share_alpha)

    # Apply Defensive Penalty
    DEF_FACTOR = 0.5 
    penalty_alpha = def_beta * DEF_FACTOR
    penalty_beta = def_alpha * DEF_FACTOR
    
    pts_alpha_base = expected_pts_alpha - penalty_alpha
    pts_beta_base = expected_pts_beta - penalty_beta
    
    # Introduce Randomness
    random_shift = np.random.normal(loc=0, scale=8.0) 

    # Final score: add/subtract random shift
    final_pts_alpha = round(max(0, pts_alpha_base + random_shift))
    final_pts_beta = round(max(0, pts_beta_base - random_shift)) 

    return final_pts_alpha, final_pts_beta


# --- 4. Main App Execution ---

st.set_page_config(layout="wide", page_title="NBA Team Optimizer")

st.title("🏀 Head-to-Head Custom Team Optimizer")
st.write("Define two unique strategies (Team Alpha and Team Beta) and compare their optimal lineups side-by-side.")

# --- Load Data and AI Model ---
player_pool_df = load_data('database_24_25.csv')
input_features = 6 
# Load the AI Model here (requires nba_model.pth)
ai_model = load_model('nba_model.pth', input_features) 

# --- STRATEGY SECTION (Dual Custom Inputs) ---
with st.expander("Step 1: Define Head-to-Head Strategies (Max Total 100)", expanded=True):
    
    col_alpha, col_beta = st.columns(2)

    # --- TEAM ALPHA (Strategy 1) ---
    with col_alpha:
        st.subheader("Strategy 1: Team Alpha 🟢 (Max 100 pts)")
        alpha_c1, alpha_c2, alpha_c3 = st.columns(3)
        
        with alpha_c1:
            w1_pts = st.number_input("Points (PTS)", min_value=0, max_value=100, value=25, key='a_pts', step=5)
            w1_stl = st.number_input("Steals (STL)", min_value=0, max_value=100, value=15, key='a_stl', step=5)
        with alpha_c2:
            w1_ast = st.number_input("Assists (AST)", min_value=0, max_value=100, value=15, key='a_ast', step=5)
            w1_blk = st.number_input("Blocks (BLK)", min_value=0, max_value=100, value=15, key='a_blk', step=5)
        with alpha_c3:
            w1_trb = st.number_input("Rebounds (TRB)", min_value=0, max_value=100, value=15, key='a_trb', step=5)
            w1_fg  = st.number_input("Efficiency (FG%)", min_value=0, max_value=100, value=15, key='a_fg', step=5)

        total_weight_1 = w1_pts + w1_ast + w1_trb + w1_stl + w1_blk + w1_fg
        
        if total_weight_1 > 100:
            st.error(f"Alpha Budget Exceeded: {total_weight_1}/100")
            valid_alpha = False
        else:
            st.success(f"Alpha Budget Used: {total_weight_1}/100")
            valid_alpha = True

        custom_weights_1 = {k: v / total_weight_1 for k, v in zip(['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%'], [w1_pts, w1_ast, w1_trb, w1_stl, w1_blk, w1_fg])}
            
        st.caption("Alpha Strategy Breakdown:")
        strategy_df_1 = pd.DataFrame([custom_weights_1]).T
        strategy_df_1.columns = ["Importance"]
        st.bar_chart(strategy_df_1, height=150)

    # --- TEAM BETA (Strategy 2) ---
    with col_beta:
        st.subheader("Strategy 2: Team Beta 🔵 (Max 100 pts)")
        beta_c1, beta_c2, beta_c3 = st.columns(3)

        with beta_c1:
            w2_pts = st.number_input("Points (PTS)", min_value=0, max_value=100, value=20, key='b_pts', step=5)
            w2_stl = st.number_input("Steals (STL)", min_value=0, max_value=100, value=25, key='b_stl', step=5)
        with beta_c2:
            w2_ast = st.number_input("Assists (AST)", min_value=0, max_value=100, value=10, key='b_ast', step=5)
            w2_blk = st.number_input("Blocks (BLK)", min_value=0, max_value=100, value=25, key='b_blk', step=5)
        with beta_c3:
            w2_trb = st.number_input("Rebounds (TRB)", min_value=0, max_value=100, value=10, key='b_trb', step=5)
            w2_fg  = st.number_input("Efficiency (FG%)", min_value=0, max_value=100, value=10, key='b_fg', step=5)
        
        total_weight_2 = w2_pts + w2_ast + w2_trb + w2_stl + w2_blk + w2_fg

        if total_weight_2 > 100:
            st.error(f"Beta Budget Exceeded: {total_weight_2}/100")
            valid_beta = False
        else:
            st.success(f"Beta Budget Used: {total_weight_2}/100")
            valid_beta = True

        custom_weights_2 = {k: v / total_weight_2 for k, v in zip(['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%'], [w2_pts, w2_ast, w2_trb, w2_stl, w2_blk, w2_fg])}
            
        st.caption("Beta Strategy Breakdown:")
        strategy_df_2 = pd.DataFrame([custom_weights_2]).T
        strategy_df_2.columns = ["Importance"]
        st.bar_chart(strategy_df_2, height=150)


# --- RESULTS SECTION ---
st.divider()
st.subheader("Step 2: Head-to-Head Lineup Comparison")

tab1, tab2, tab3 = st.tabs(["🏆 Compare Lineups", "🤖 ANN Model Details", "📋 Full Player Pool"])

with tab1:
    if player_pool_df is not None and valid_alpha and valid_beta:
        if st.button("Run Head-to-Head Comparison", type="primary", use_container_width=True):
            
            # --- Perform Both Searches ---
            with st.spinner('Calculating both optimal teams...'):
                team_alpha, score_alpha, time_alpha = find_optimal_team_custom(player_pool_df, custom_weights_1, label="Team Alpha")
                team_beta, score_beta, time_beta = find_optimal_team_custom(player_pool_df, custom_weights_2, label="Team Beta")

            # --- Run Single Game Simulation ---
            if team_alpha is not None and team_beta is not None:
                st.markdown("### 🏀 Single Game Simulation (105 Possessions)")
                
                # Get the AI's opinion on the custom teams
                ai_score_alpha = get_ai_score(team_alpha, ai_model, player_pool_df)
                ai_score_beta = get_ai_score(team_beta, ai_model, player_pool_df)

                with st.spinner('Running game simulation...'):
                    pts_alpha, pts_beta = run_single_game(team_alpha, team_beta, score_alpha, score_beta, possessions=105)
                    
                    winner = "Team Alpha 🟢" if pts_alpha > pts_beta else ("Team Beta 🔵" if pts_beta > pts_alpha else "Tie")
                    
                    sim_col1, sim_col2, sim_col3 = st.columns(3)
                    
                    sim_col1.metric("Final Score", f"{pts_alpha} - {pts_beta}")
                    sim_col2.metric("Winning Team", winner)
                    
                    if pts_alpha != pts_beta:
                        sim_col3.metric("Winning Margin", f"{abs(pts_alpha - pts_beta)} pts")
                    else:
                        sim_col3.metric("Winning Margin", "0 pts (Tie)")

                    if winner == "Tie":
                        st.info(f"**Final Result:** The game ended in a tie! {pts_alpha} - {pts_beta}.")
                    else:
                        st.success(f"**Final Result:** {winner} wins {pts_alpha} - {pts_beta}!")
                    st.divider()

            # --- Display Side-by-Side Results ---
            col_res_alpha, col_res_beta = st.columns(2)
            
            def display_team_results(col, team_df, score, ai_score, strategy_label, run_time, color):
                if team_df is not None:
                    with col:
                        st.markdown(f"### {strategy_label} {color}")
                        
                        # Highlighted AI Metric
                        st.markdown("**AI Model Opinion**")
                        ai_c1, ai_c2 = st.columns(2)
                        ai_c1.metric("Your Strategy Score", f"{score:.3f}")
                        ai_c2.metric("ANN Predicted Score", f"{ai_score:.3f}")

                        st.caption(f"Search time: {run_time:.2f} seconds")
                        
                        # Display the team with formatting
                        st.dataframe(
                            team_df[['Player', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%']].style.format({
                                'PTS': "{:.1f}", 'AST': "{:.1f}", 'TRB': "{:.1f}", 
                                'STL': "{:.1f}", 'BLK': "{:.1f}", 'FG%': "{:.3f}"
                            }), 
                            use_container_width=True
                        )
                        
                        st.markdown("#### Combined Totals")
                        totals = team_df[['PTS', 'AST', 'TRB', 'STL', 'BLK']].sum()
                        avg_fg = team_df['FG%'].mean()
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total PTS", f"{totals['PTS']:.0f}")
                        m2.metric("Total AST", f"{totals['AST']:.0f}")
                        m3.metric("Total TRB", f"{totals['TRB']:.0f}")
                        
                        d1, d2 = st.columns(2)
                        d1.metric("Defense (STL+BLK)", f"{totals['STL']+totals['BLK']:.1f}")
                        d2.metric("Avg FG%", f"{avg_fg:.3f}")
                else:
                    with col:
                        st.error("Team calculation failed.")


            # Call helper function for each column
            display_team_results(col_res_alpha, team_alpha, score_alpha, ai_score_alpha, "Team Alpha", time_alpha, "🟢")
            display_team_results(col_res_beta, team_beta, score_beta, ai_score_beta, "Team Beta", time_beta, "🔵")

    elif player_pool_df is None:
        st.error("Cannot run comparison. Player data could not be loaded. Check your file path!")
    elif not valid_alpha or not valid_beta:
        st.warning("Please adjust your number inputs. Both Team Alpha and Team Beta must use a total weight of 100 or less.")

# --- ANN DETAILS TAB ---
with tab2:
    st.header("🤖 Artificial Neural Network (ANN) Architecture")
    st.write("The pre-trained PyTorch model provides a fixed definition of an 'Optimal Team' based on historical data. This score is used as a baseline for comparison (AI Commentary).")
    st.markdown("**Core Architecture (Multilayer Perceptron - MLP):**")
    st.code("""
    class MLP(nn.Module):
        def __init__(self, input_size):
            # Input size is 6 (sum of 6 player stats)
            super(MLP, self).__init__()
            self.layer1 = nn.Linear(input_size, 64)
            self.layer2 = nn.Linear(64, 32)
            self.dropout = nn.Dropout(0.2)
            self.output_layer = nn.Linear(32, 1) # Outputs the single Team Score
            self.relu = nn.ReLU()
    """, language='python')
    
    st.markdown("""
    **How the AI differs from Your Strategy:**
    * **Your Strategy (Heuristic):** Uses the dynamic weights (sliders) you define to find the optimal lineup.
    * **ANN Score (AI Commentary):** Uses fixed, learned weights from its training to predict how *it* would score your team. If the scores differ greatly, it means your strategy prioritizes different aspects (like Defense) than the historical data the AI learned from.
    """)

# --- PLAYER POOL TAB ---
with tab3:
    if player_pool_df is not None:
        st.header(f"Available Players ({len(player_pool_df)})")
        st.dataframe(player_pool_df, use_container_width=True)