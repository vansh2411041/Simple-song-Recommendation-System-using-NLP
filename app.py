import pandas as pd
import streamlit as st
import os
from nlp_utils import SongAnalyzer
import plotly.express as px

def load_and_merge_data(data_folder="data"):
    """Load all artist CSVs with proper error handling"""
    try:
        all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        if not all_files:
            st.error("No CSV files found in the data folder!")
            return None
            
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(os.path.join(data_folder, file))
                # Clean and standardize columns
                df = df.rename(columns={
                    'Artist': 'artist',
                    'Title': 'name',
                    'Lyric': 'lyrics',
                    'Album': 'album',
                    'Year': 'year',
                    'Date': 'release_date'
                })
                df = df.dropna(subset=['name', 'lyrics'])
                df['name'] = df['name'].str.strip()
                df['artist'] = df['artist'].str.strip()
                dfs.append(df)
            except Exception as e:
                st.warning(f"Couldn't process {file}: {str(e)}")
                continue
                
        return pd.concat(dfs, ignore_index=True) if dfs else None
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

def display_sentiment_results(results):
    """Visualize results from all models"""
    st.subheader("Multi-Model Sentiment Analysis")
    
    with st.expander("TextBlob Analysis"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Polarity", f"{results['textblob']['polarity']:.2f}",
                     help="-1 (Negative) to +1 (Positive)")
        with col2:
            st.metric("Subjectivity", f"{results['textblob']['subjectivity']:.2f}",
                     help="0 (Objective) to 1 (Subjective)")
    
    with st.expander("VADER Analysis"):
        fig = px.bar(
            x=["Positive", "Negative", "Neutral"],
            y=[results['vader']['positive'], results['vader']['negative'], results['vader']['neutral']],
            labels={'x': 'Sentiment', 'y': 'Score'},
            title="VADER Sentiment Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Compound Score", f"{results['vader']['compound']:.2f}",
                 help="-1 (Extreme Negative) to +1 (Extreme Positive)")
    
    with st.expander("Transformers Analysis"):
        label = results['transformers']['label']
        score = results['transformers']['score']
        st.metric("Sentiment", f"{label.title()} ({score:.2%})")

def main():
    st.title("🎵 Advanced Song Analysis")
    analyzer = SongAnalyzer()
    
    # Load data
    if 'merged_df' not in st.session_state:
        with st.spinner("Loading song database..."):
            st.session_state.merged_df = load_and_merge_data()
    
    if st.session_state.merged_df is None:
        st.stop()
    
    df = st.session_state.merged_df
    st.success(f"Loaded {len(df)} songs from {df['artist'].nunique()} artists")
    
    # Artist selection
    artists = sorted(df['artist'].unique())
    selected_artist = st.selectbox("Filter by artist", artists, index=0)
    
    # Song selection
    filtered_songs = df[df['artist'] == selected_artist]
    song_names = filtered_songs['name'].dropna().unique()
    selected_song = st.selectbox("Choose a song", song_names, index=0)
    
    if st.button("Analyze & Recommend"):
        selected_row = filtered_songs[filtered_songs['name'] == selected_song]
        
        if len(selected_row) == 0:
            st.error("Song not found!")
            return
            
        lyrics = selected_row.iloc[0]['lyrics']
        st.subheader(f"Analysis for: {selected_song}")
        
        with st.expander("View Lyrics"):
            st.text_area("Lyrics", lyrics, height=200)
        
        # Run analysis
        with st.spinner("Running advanced NLP analysis..."):
            analysis_results = analyzer.comprehensive_analysis(lyrics)
            display_sentiment_results(analysis_results)
        
        # Recommendations
        with st.spinner("Finding similar songs..."):
            similarity_matrix = analyzer.find_similar_songs(df["lyrics"])
            similar_indices = similarity_matrix[selected_row.index[0]].argsort()[-6:-1][::-1]
            recommendations = df.iloc[similar_indices][['name', 'artist']]
            
            st.subheader("🎧 Recommended Similar Songs")
            st.dataframe(recommendations.reset_index(drop=True))

if __name__ == "__main__":
    main()