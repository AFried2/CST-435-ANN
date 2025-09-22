import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# --- 1. PyTorch Model Definition (Copied from your notebook) ---
# This class MUST be defined so we can load the saved model weights.
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

# --- 2. Caching and Loading Functions ---
# @st.cache_data is a powerful Streamlit feature that prevents reloading data
# and re-running computations on every user interaction.

@st.cache_data
def load_data(filepath):
    """Loads and processes the player data, returning a DataFrame of season averages."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please make sure it's in your GitHub repository.")
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
        st.error(f"Error: The model file '{model_path}' was not found. Please make sure you have saved and uploaded it.")
        return None
    model.eval()  # Set the model to evaluation mode
    return model

@st.cache_data
def find_optimal_team(_model, player_pool):
    """Searches for the optimal team and returns the team DataFrame and score."""
    if _model is None or player_pool is None:
        return None, None

    features = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%']
    
    # Scale the data for the model
    scaler = MinMaxScaler()
    player_pool_scaled = player_pool.copy()
    player_pool_scaled[features] = scaler.fit_transform(player_pool_scaled[features])

    best_team_indices = None
    best_predicted_score = -np.inf
    search_space_size = 200000  # You can adjust this for speed vs. accuracy
    player_ids = player_pool.index.to_numpy()

    progress_bar = st.progress(0, text="Searching for the optimal team...")

    for i in range(search_space_size):
        team_indices = np.random.choice(player_ids, 5, replace=False)
        team_df_scaled = player_pool_scaled.loc[team_indices]
        input_vector = team_df_scaled[features].sum().values
        
        with torch.no_grad():
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).view(1, -1)
            predicted_score = _model(input_tensor).item()

        if predicted_score > best_predicted_score:
            best_predicted_score = predicted_score
            best_team_indices = team_indices
        
        # Update progress bar every 1000 iterations
        if (i + 1) % 1000 == 0:
            progress_bar.progress(int((i + 1) / search_space_size * 100), text=f"Searching... {i+1}/{search_space_size} combinations tested.")

    progress_bar.empty() # Clear the progress bar when done
    optimal_team_df = player_pool.loc[best_team_indices]
    return optimal_team_df, best_predicted_score


# --- 3. Streamlit App UI ---
st.set_page_config(layout="wide")

st.title("🏀 NBA Optimal Team Finder using an ANN")
st.write("This application uses a pre-trained Artificial Neural Network (PyTorch) to identify the optimal 5-player team from a dataset of player season averages.")

# --- Load Data and Model ---
player_pool_df = load_data('database_24_25.csv')
input_features = 6 # Corresponds to the 6 features we are using
model = load_model('nba_model.pth', input_features)


# --- UI Tabs ---
tab1, tab2, tab3 = st.tabs(["🏆 The Optimal Team", " Player Pool", "🤖 Model Details"])

with tab1:
    st.header("The AI-Selected Optimal Team")
    if model and player_pool_df is not None:
        # Using a button to trigger the search
        if st.button("Find the Optimal Team"):
            with st.spinner('Calculating... This may take a minute.'):
                optimal_team, team_score = find_optimal_team(model, player_pool_df)
                if optimal_team is not None:
                    st.metric(label="Predicted Team Score", value=f"{team_score:.2f}")
                    st.dataframe(optimal_team.round(2))
                else:
                    st.warning("Could not find an optimal team. Please check the data and model files.")
    else:
        st.warning("Data or model files could not be loaded. Please check the setup.")


with tab2:
    st.header(f"Full Player Pool ({len(player_pool_df) if player_pool_df is not None else 0} Players)")
    st.write("This table shows the season average statistics for every player in the dataset.")
    if player_pool_df is not None:
        st.dataframe(player_pool_df.round(2))

with tab3:
    st.header("Model Architecture")
    st.write("The model is a simple Multilayer Perceptron (MLP) built with PyTorch. It takes the combined stats of a 5-player team as input and outputs a single 'Team Score'. The architecture is as follows:")
    st.code("""
    class MLP(nn.Module):
        def __init__(self, input_size):
            super(MLP, self).__init__()
            self.layer1 = nn.Linear(input_size, 64)
            self.layer2 = nn.Linear(64, 32)
            self.dropout = nn.Dropout(0.2)
            self.output_layer = nn.Linear(32, 1)
            self.relu = nn.ReLU()
    """, language='python')
