#!/usr/bin/env python3
"""
Test script for the school dashboard functionality
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def generate_sample_data(n_schools=50):
    """Generate sample school data matching the user's column structure"""
    np.random.seed(42)
    
    # Define realistic options
    regions = ['North', 'South', 'East', 'West', 'Central']
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
    channels = ['Direct', 'Partner', 'Online', 'Referral']
    boards = ['CBSE', 'ICSE', 'State Board', 'IB', 'Cambridge']
    programs = ['STEM', 'FinLit', 'Comms', 'STEM+FinLit']
    grades = ['1-5', '6-8', '9-10', '11-12']
    fee_ranges = ['<50k', '50k-1L', '1L-2L', '2L-5L', '>5L']
    strength_ranges = ['<200', '200-500', '500-1000', '1000-2000', '>2000']
    categories = ['Premium', 'Mid-tier', 'Budget', 'Elite']
    onboarding_type = ['Full year', 'Mid year']
    
    data = []
    for i in range(n_schools):
        region = np.random.choice(regions)
        city = np.random.choice(cities)
        board = np.random.choice(boards)
        
        # Correlate fees with board type
        if board in ['IB', 'Cambridge']:
            fee_range = np.random.choice(['2L-5L', '>5L'], p=[0.3, 0.7])
            category = np.random.choice(['Premium', 'Elite'], p=[0.4, 0.6])
        elif board == 'ICSE':
            fee_range = np.random.choice(['1L-2L', '2L-5L'], p=[0.6, 0.4])
            category = np.random.choice(['Mid-tier', 'Premium'], p=[0.6, 0.4])
        else:
            fee_range = np.random.choice(['<50k', '50k-1L', '1L-2L'], p=[0.4, 0.4, 0.2])
            category = np.random.choice(['Budget', 'Mid-tier'], p=[0.6, 0.4])
        
        strength = np.random.randint(100, 1500)
        if strength < 200:
            strength_range = '<200'
        elif strength < 500:
            strength_range = '200-500'
        elif strength < 1000:
            strength_range = '500-1000'
        else:
            strength_range = '1000-2000'
        
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
            'Month od school onboarding': np.random.choice(['April', 'May', 'June']),
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

def perform_school_segmentation(df):
    """Perform clustering to segment schools"""
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
    
    return features_for_clustering

def generate_curriculum_recommendations(df):
    """Generate personalized curriculum recommendations"""
    recommendations = {}
    
    for segment in df['Segment_Label'].unique():
        segment_data = df[df['Segment_Label'] == segment]
        
        # Analyze segment characteristics
        avg_strength = segment_data['Strength'].mean()
        common_board = segment_data['Board'].mode()[0] if len(segment_data) > 0 else 'CBSE'
        common_fee_range = segment_data['Fees Range'].mode()[0] if len(segment_data) > 0 else '<50k'
        
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
                'fee_range': common_fee_range
            },
            'curriculum': {
                'STEM': stem_rec,
                'FinLit': finlit_rec,
                'Comms': comms_rec
            }
        }
    
    return recommendations

def main():
    print("🏫 Testing School Analysis Dashboard")
    print("=" * 50)
    
    # Generate sample data
    print("1. Generating sample data...")
    df = generate_sample_data(100)
    print(f"   Generated {len(df)} schools")
    print(f"   Columns: {list(df.columns)}")
    
    # Basic statistics
    print("\n2. Basic Statistics:")
    print(f"   Total Students: {df['Strength'].sum():,}")
    print(f"   Average Revenue/Student: ₹{df['Rev/ student'].mean():,.0f}")
    print(f"   Total Business: ₹{df['Total Business'].sum()/10000000:.1f} Cr")
    
    # Board distribution
    print("\n3. Board Distribution:")
    print(df['Board'].value_counts())
    
    # Category distribution
    print("\n4. Category Distribution:")
    print(df['Category'].value_counts())
    
    # Perform segmentation
    print("\n5. Performing school segmentation...")
    df_segmented = perform_school_segmentation(df)
    print("   Segmentation completed!")
    
    # Segment summary
    print("\n6. Segment Summary:")
    segment_summary = df_segmented.groupby('Segment_Label').agg({
        'Strength': ['count', 'mean'],
        'Rev/ student': 'mean',
        'Total Business': 'sum'
    }).round(0)
    segment_summary.columns = ['School Count', 'Avg Strength', 'Avg Rev/Student', 'Total Business']
    print(segment_summary)
    
    # Generate recommendations
    print("\n7. Generating curriculum recommendations...")
    recommendations = generate_curriculum_recommendations(df_segmented)
    
    print("\n8. Curriculum Recommendations by Segment:")
    for segment, rec_data in recommendations.items():
        print(f"\n   🎯 {segment}")
        print(f"      Schools: {rec_data['characteristics']['avg_strength']} avg strength")
        print(f"      Common Board: {rec_data['characteristics']['common_board']}")
        print(f"      Fee Range: {rec_data['characteristics']['fee_range']}")
        print(f"      STEM: {rec_data['curriculum']['STEM']}")
        print(f"      FinLit: {rec_data['curriculum']['FinLit']}")
        print(f"      Comms: {rec_data['curriculum']['Comms']}")
    
    print("\n✅ Dashboard test completed successfully!")
    print("\nTo run the full dashboard:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run dashboard: streamlit run school_dashboard.py")
    
    # Save sample data for testing
    df.to_csv('sample_school_data.csv', index=False)
    print(f"\n💾 Sample data saved as 'sample_school_data.csv'")

if __name__ == "__main__":
    main()