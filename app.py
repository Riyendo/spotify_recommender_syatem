import streamlit as st
import pandas as pd
from numpy import load
from scipy.sparse import load_npz

from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_data():
    songs_data = pd.read_csv("data/cleaned_data.csv")
    filtered_data = pd.read_csv("data/collab_filtered_data.csv")
    track_ids = load("data/track_ids.npy", allow_pickle=True)
    transformed_data = load_npz("data/transformed_data.npz")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")
    return (songs_data, filtered_data, track_ids,
            transformed_data, interaction_matrix, transformed_hybrid_data)


(songs_data, filtered_data, track_ids,
 transformed_data, interaction_matrix, transformed_hybrid_data) = load_data()


# --------------------------------------------------------------------- helpers
def song_exists(df, song, artist):
    return ((df["name"] == song) & (df["artist"] == artist)).any()


def show_recommendations(recommendations):
    for ind, rec in recommendations.iterrows():
        title = rec["name"].title()
        artist = rec["artist"].title()

        if ind == 0:
            st.markdown("## Currently Playing")
            st.markdown(f"#### **{title}** by **{artist}**")
        else:
            if ind == 1:
                st.markdown("### Next Up 🎵")
            st.markdown(f"#### {ind}. **{title}** by **{artist}**")

        st.audio(rec["spotify_preview_url"])
        st.write("---")


def run_hybrid(song, artist, k, content_weight):
    recommender = HybridRecommenderSystem(
        number_of_recommendations=k,
        weight_content_based=content_weight,
    )
    return recommender.give_recommendations(
        song_name=song,
        artist_name=artist,
        songs_data=filtered_data,
        transformed_matrix=transformed_hybrid_data,
        track_ids=track_ids,
        interaction_matrix=interaction_matrix,
    )


# ------------------------------------------------------------------------- UI
st.title("Welcome to the Spotify Song Recommender!")
st.write("### Enter a song and pick a recommender 🎵🎧")

song_name = st.text_input("Song name:").strip().lower()
artist_name = st.text_input("Artist name:").strip().lower()
k = st.selectbox("How many recommendations?", [5, 10, 15, 20], index=1)

method = st.radio(
    "Recommendation method",
    ["Content-Based", "Collaborative", "Hybrid"],
    horizontal=True,
    help=(
        "Content-Based: songs with similar audio/metadata features.\n\n"
        "Collaborative: songs that similar listeners enjoyed.\n\n"
        "Hybrid: a mix of both (you control the balance)."
    ),
)

# Hybrid needs a diversity slider
content_weight = None
if method == "Hybrid":
    diversity = st.slider("Diversity in Recommendations", 1, 9, 5)
    content_weight = 1 - diversity / 10
    st.bar_chart(
        pd.DataFrame({
            "type": ["Personalized", "Diverse"],
            "ratio": [10 - diversity, diversity],
        }),
        x="type", y="ratio",
    )
elif method == "Collaborative":
    content_weight = 0.0  # collaborative signal only

if st.button("Get Recommendations"):
    if not song_name or not artist_name:
        st.warning("Please enter both a song name and an artist name.")
        st.stop()

    # Collaborative / Hybrid only work for songs in the filtered dataset
    data_to_check = songs_data if method == "Content-Based" else filtered_data
    if not song_exists(data_to_check, song_name, artist_name):
        if method == "Content-Based":
            st.error(f"Sorry, we couldn't find {song_name} by {artist_name} in our database.")
        else:
            st.error(
                f"{song_name} by {artist_name} isn't available for the {method} method. "
                "Try Content-Based instead."
            )
        st.stop()

    st.write("Recommendations for", f"**{song_name}** by **{artist_name}**")

    if method == "Content-Based":
        recs = content_recommendation(
            song_name=song_name,
            artist_name=artist_name,
            songs_data=songs_data,
            transformed_data=transformed_data,
            k=k,
        )
    else:
        recs = run_hybrid(song_name, artist_name, k, content_weight)

    show_recommendations(recs)