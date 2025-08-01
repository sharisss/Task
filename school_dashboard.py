import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="School Analysis Dashboard",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data(n_schools=500):
    """Generate sample school data matching the user's column structure"""
    np.random.seed(42)
    
    # Define realistic options
    regions = ['North', 'South', 'East', 'West', 'Central']
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
              'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur']
    channels = ['Direct', 'Partner', 'Online', 'Referral']
    boards = ['CBSE', 'ICSE', 'State Board', 'IB', 'Cambridge']
    programs = ['STEM', 'FinLit', 'Comms', 'STEM+FinLit', 'STEM+Comms', 'FinLit+Comms', 'All Three']
    grades = ['1-5', '6-8', '9-10', '11-12', '1-8', '6-12', '1-12']
    fee_ranges = ['<50k', '50k-1L', '1L-2L', '2L-5L', '>5L']
    strength_ranges = ['<200', '200-500', '500-1000', '1000-2000', '>2000']
    categories = ['Premium', 'Mid-tier', 'Budget', 'Elite']
    onboarding_type = ['Full year', 'Mid year']
    
    data = []
    for i in range(n_schools):
        # Generate correlated data for realistic patterns
        region = np.random.choice(regions)
        city = np.random.choice(cities)
        board = np.random.choice(boards)
        
        # Correlate fees with board type
        if board in ['IB', 'Cambridge']:
            fee_range = np.random.choice(['2L-5L', '>5L'], p=[0.3, 0.7])
            category = np.random.choice(['Premium', 'Elite'], p=[0.4, 0.6])
        elif board == 'ICSE':
            fee_range = np.random.choice(['1L-2L', '2L-5L', '>5L'], p=[0.3, 0.5, 0.2])
            category = np.random.choice(['Mid-tier', 'Premium'], p=[0.6, 0.4])
        elif board == 'CBSE':
            fee_range = np.random.choice(['<50k', '50k-1L', '1L-2L', '2L-5L'], p=[0.2, 0.4, 0.3, 0.1])
            category = np.random.choice(['Budget', 'Mid-tier', 'Premium'], p=[0.4, 0.5, 0.1])
        else:  # State Board
            fee_range = np.random.choice(['<50k', '50k-1L', '1L-2L'], p=[0.5, 0.4, 0.1])
            category = np.random.choice(['Budget', 'Mid-tier'], p=[0.7, 0.3])
        
        # Generate strength based on category
        if category == 'Elite':
            strength = np.random.randint(300, 800)
            strength_range = '200-500' if strength < 500 else '500-1000'
        elif category == 'Premium':
            strength = np.random.randint(400, 1200)
            if strength < 500:
                strength_range = '200-500'
            elif strength < 1000:
                strength_range = '500-1000'
            else:
                strength_range = '1000-2000'
        elif category == 'Mid-tier':
            strength = np.random.randint(200, 1500)
            if strength < 200:
                strength_range = '<200'
            elif strength < 500:
                strength_range = '200-500'
            elif strength < 1000:
                strength_range = '500-1000'
            else:
                strength_range = '1000-2000'
        else:  # Budget
            strength = np.random.randint(100, 800)
            if strength < 200:
                strength_range = '<200'
            elif strength < 500:
                strength_range = '200-500'
            else:
                strength_range = '500-1000'
        
        # Revenue per student based on fee range
        rev_per_student = {
            '<50k': np.random.randint(20000, 45000),
            '50k-1L': np.random.randint(50000, 95000),
            '1L-2L': np.random.randint(100000, 190000),
            '2L-5L': np.random.randint(200000, 450000),
            '>5L': np.random.randint(500000, 1000000)
        }[fee_range]
        
        total_business = strength * rev_per_student
        
        school_data = {
            'No.': i + 1,
            'Academic Year': np.random.choice(['2023-24', '2024-25']),
            'Month od school onboarding': np.random.choice(['April', 'May', 'June', 'July', 'August', 'September']),
            'Region': region,
            'City': city,
            'Channel': np.random.choice(channels),
            'POC': f'POC_{i+1}',
            'School Name': f'School_{i+1}',
            'School and Program': f'School_{i+1}_{np.random.choice(programs)}',
            'Program': np.random.choice(programs),
            'Grade': np.random.choice(grades),
            'Board': board,
            'Fees Range': fee_range,
            'Strength Range': strength_range,
            'Strength': strength,
            'Rev/ student': rev_per_student,
            'Sum Product': total_business,
            'Total Business': total_business,
            'Category': category,
            'Full year/ mid year': np.random.choice(onboarding_type)
        }
        data.append(school_data)
    
    return pd.DataFrame(data)

@st.cache_data
def load_data(uploaded_file=None):
    """Load data from uploaded file or generate sample data"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return generate_sample_data()
    else:
        return generate_sample_data()

def perform_school_segmentation(df):
    """Perform clustering to segment schools"""
    # Prepare features for clustering
    features_for_clustering = df.copy()
    
    # Encode categorical variables
    le_board = LabelEncoder()
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    
    features_for_clustering['Board_encoded'] = le_board.fit_transform(df['Board'])
    features_for_clustering['Category_encoded'] = le_category.fit_transform(df['Category'])
    features_for_clustering['Region_encoded'] = le_region.fit_transform(df['Region'])
    
    # Select numerical features
    clustering_features = ['Strength', 'Rev/ student', 'Board_encoded', 'Category_encoded', 'Region_encoded']
    X = features_for_clustering[clustering_features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    features_for_clustering['School_Segment'] = kmeans.fit_predict(X_scaled)
    
    # Add segment labels
    segment_labels = {
        0: 'Budget Local Schools',
        1: 'Mid-tier Regional Schools', 
        2: 'Premium Urban Schools',
        3: 'Elite International Schools',
        4: 'Large State Board Schools'
    }
    
    features_for_clustering['Segment_Label'] = features_for_clustering['School_Segment'].map(segment_labels)
    
    return features_for_clustering, scaler, kmeans

def generate_curriculum_recommendations(df):
    """Generate personalized curriculum recommendations"""
    recommendations = {}
    
    for segment in df['Segment_Label'].unique():
        segment_data = df[df['Segment_Label'] == segment]
        
        # Analyze segment characteristics
        avg_strength = segment_data['Strength'].mean()
        common_board = segment_data['Board'].mode()[0]
        common_fee_range = segment_data['Fees Range'].mode()[0]
        common_programs = segment_data['Program'].value_counts().head(3)
        
        # Generate recommendations based on segment
        if 'Elite' in segment or 'Premium' in segment:
            stem_rec = "Advanced STEM with robotics, AI, and coding projects"
            finlit_rec = "Investment basics, entrepreneurship, and financial planning"
            comms_rec = "Public speaking, debate, and international communication"
        elif 'Mid-tier' in segment:
            stem_rec = "Practical STEM with experiments and basic programming"
            finlit_rec = "Personal finance, budgeting, and savings concepts"
            comms_rec = "Presentation skills and effective communication"
        else:  # Budget/Local
            stem_rec = "Foundational STEM with hands-on activities"
            finlit_rec = "Basic financial literacy and money management"
            comms_rec = "Language skills and basic communication"
        
        recommendations[segment] = {
            'characteristics': {
                'avg_strength': int(avg_strength),
                'common_board': common_board,
                'fee_range': common_fee_range,
                'popular_programs': common_programs.to_dict()
            },
            'curriculum': {
                'STEM': stem_rec,
                'FinLit': finlit_rec,
                'Comms': comms_rec
            }
        }
    
    return recommendations

def main():
    st.markdown('<h1 class="main-header">🏫 School Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze school types and personalize curriculum for STEM, FinLit, and Communication programs")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your school dataset (CSV)", 
        type=['csv'],
        help="Upload a CSV file with your school data, or use the sample data below"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if uploaded_file is None:
        st.sidebar.info("Using sample data. Upload your CSV file to analyze your actual data.")
    
    # Perform segmentation
    df_segmented, scaler, kmeans = perform_school_segmentation(df)
    
    # Generate recommendations
    recommendations = generate_curriculum_recommendations(df_segmented)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🎯 School Segments", 
        "📚 Curriculum Recommendations", 
        "📊 Detailed Analysis",
        "🔍 School Finder"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Schools", len(df))
        with col2:
            st.metric("Total Students", f"{df['Strength'].sum():,}")
        with col3:
            st.metric("Avg Revenue/Student", f"₹{df['Rev/ student'].mean():,.0f}")
        with col4:
            st.metric("Total Business", f"₹{df['Total Business'].sum()/10000000:.1f}Cr")
        
        # Key distributions
        col1, col2 = st.columns(2)
        
        with col1:
            fig_board = px.pie(df, names='Board', title='Distribution by Board')
            st.plotly_chart(fig_board, use_container_width=True)
            
            fig_category = px.pie(df, names='Category', title='Distribution by Category')
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            fig_region = px.bar(df.groupby('Region').size().reset_index(name='count'), 
                               x='Region', y='count', title='Schools by Region')
            st.plotly_chart(fig_region, use_container_width=True)
            
            fig_program = px.bar(df.groupby('Program').size().reset_index(name='count'), 
                                x='Program', y='count', title='Program Popularity')
            st.plotly_chart(fig_program, use_container_width=True)
    
    with tab2:
        st.header("School Segmentation Analysis")
        
        # Segment overview
        segment_summary = df_segmented.groupby('Segment_Label').agg({
            'Strength': ['count', 'mean'],
            'Rev/ student': 'mean',
            'Total Business': 'sum'
        }).round(0)
        
        segment_summary.columns = ['School Count', 'Avg Strength', 'Avg Rev/Student', 'Total Business']
        st.dataframe(segment_summary, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(df_segmented, 
                                   x='Strength', y='Rev/ student', 
                                   color='Segment_Label',
                                   size='Total Business',
                                   hover_data=['Board', 'Category'],
                                   title='School Segments: Strength vs Revenue per Student')
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_box = px.box(df_segmented, x='Segment_Label', y='Total Business', 
                            title='Business Distribution by Segment')
            fig_box.update_xaxis(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Segment characteristics
        st.subheader("Segment Characteristics")
        for segment in df_segmented['Segment_Label'].unique():
            with st.expander(f"📋 {segment}"):
                segment_data = df_segmented[df_segmented['Segment_Label'] == segment]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Schools", len(segment_data))
                    st.write("**Top Boards:**")
                    st.write(segment_data['Board'].value_counts().head(3))
                
                with col2:
                    st.metric("Avg Strength", f"{segment_data['Strength'].mean():.0f}")
                    st.write("**Fee Ranges:**")
                    st.write(segment_data['Fees Range'].value_counts().head(3))
                
                with col3:
                    st.metric("Avg Rev/Student", f"₹{segment_data['Rev/ student'].mean():,.0f}")
                    st.write("**Popular Programs:**")
                    st.write(segment_data['Program'].value_counts().head(3))
    
    with tab3:
        st.header("Personalized Curriculum Recommendations")
        
        for segment, rec_data in recommendations.items():
            st.subheader(f"🎯 {segment}")
            
            # Segment characteristics
            chars = rec_data['characteristics']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Strength", chars['avg_strength'])
            with col2:
                st.info(f"**Common Board:** {chars['common_board']}")
            with col3:
                st.info(f"**Fee Range:** {chars['fee_range']}")
            
            # Curriculum recommendations
            st.markdown("#### 📚 Recommended Curriculum Content:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔬 STEM Lab**")
                st.markdown(f"*{rec_data['curriculum']['STEM']}*")
            
            with col2:
                st.markdown("**💰 FinLit Lab**")
                st.markdown(f"*{rec_data['curriculum']['FinLit']}*")
            
            with col3:
                st.markdown("**💬 Communication Lab**")
                st.markdown(f"*{rec_data['curriculum']['Comms']}*")
            
            st.markdown("---")
    
    with tab4:
        st.header("Detailed Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_regions = st.multiselect("Select Regions", df['Region'].unique(), default=df['Region'].unique())
        with col2:
            selected_boards = st.multiselect("Select Boards", df['Board'].unique(), default=df['Board'].unique())
        with col3:
            selected_categories = st.multiselect("Select Categories", df['Category'].unique(), default=df['Category'].unique())
        
        # Filter data
        filtered_df = df[
            (df['Region'].isin(selected_regions)) & 
            (df['Board'].isin(selected_boards)) & 
            (df['Category'].isin(selected_categories))
        ]
        
        # Analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_heatmap = px.density_heatmap(
                filtered_df, x='Fees Range', y='Board', 
                title='School Distribution: Board vs Fees'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            fig_violin = px.violin(
                filtered_df, x='Category', y='Strength',
                title='Student Strength Distribution by Category'
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with col2:
            fig_sunburst = px.sunburst(
                filtered_df, path=['Region', 'Board', 'Category'],
                values='Strength',
                title='Hierarchical View: Region → Board → Category'
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            fig_treemap = px.treemap(
                filtered_df, path=['Program', 'Board'], values='Total Business',
                title='Business Value by Program and Board'
            )
            st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Data table
        st.subheader("Filtered Data")
        st.dataframe(filtered_df, use_container_width=True)
    
    with tab5:
        st.header("School Finder & Analyzer")
        
        st.markdown("### Find similar schools and get personalized recommendations")
        
        # School selector
        school_names = df['School Name'].unique()
        selected_school = st.selectbox("Select a school to analyze:", school_names)
        
        if selected_school:
            school_data = df[df['School Name'] == selected_school].iloc[0]
            school_segment = df_segmented[df_segmented['School Name'] == selected_school]['Segment_Label'].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("School Profile")
                st.write(f"**School:** {school_data['School Name']}")
                st.write(f"**Region:** {school_data['Region']}")
                st.write(f"**City:** {school_data['City']}")
                st.write(f"**Board:** {school_data['Board']}")
                st.write(f"**Category:** {school_data['Category']}")
                st.write(f"**Strength:** {school_data['Strength']}")
                st.write(f"**Fee Range:** {school_data['Fees Range']}")
                st.write(f"**Current Programs:** {school_data['Program']}")
                st.write(f"**Segment:** {school_segment}")
            
            with col2:
                st.subheader("Recommended Curriculum")
                if school_segment in recommendations:
                    rec = recommendations[school_segment]['curriculum']
                    st.markdown("**🔬 STEM Lab:**")
                    st.info(rec['STEM'])
                    st.markdown("**💰 FinLit Lab:**")
                    st.info(rec['FinLit'])
                    st.markdown("**💬 Communication Lab:**")
                    st.info(rec['Comms'])
            
            # Similar schools
            st.subheader("Similar Schools")
            similar_schools = df_segmented[
                (df_segmented['Segment_Label'] == school_segment) & 
                (df_segmented['School Name'] != selected_school)
            ].head(10)
            
            st.dataframe(
                similar_schools[['School Name', 'Region', 'City', 'Board', 'Category', 'Strength', 'Fees Range']],
                use_container_width=True
            )

if __name__ == "__main__":
    main()