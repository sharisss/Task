# 🏫 School Analysis Dashboard

A comprehensive dashboard for analyzing school data and providing personalized curriculum recommendations for STEM, Financial Literacy, and Communication programs.

## Features

### 📊 **Overview Tab**
- Key metrics: Total schools, students, revenue, and business value
- Distribution charts for boards, categories, regions, and programs
- Quick insights into your school portfolio

### 🎯 **School Segments Tab**
- Automated school segmentation using machine learning clustering
- 5 distinct segments: Budget Local, Mid-tier Regional, Premium Urban, Elite International, Large State Board
- Visual analysis of segments with scatter plots and box plots
- Detailed characteristics for each segment

### 📚 **Curriculum Recommendations Tab**
- Personalized curriculum content for each school segment
- Tailored recommendations for STEM, FinLit, and Communication labs
- Based on school characteristics: board type, fee structure, student strength

### 📈 **Detailed Analysis Tab**
- Interactive filters by region, board, and category
- Advanced visualizations: heatmaps, violin plots, sunburst charts, treemaps
- Filterable data table for detailed exploration

### 🔍 **School Finder Tab**
- Individual school analysis and profiling
- Personalized curriculum recommendations for specific schools
- Similar school finder based on segmentation

## Data Structure

Your CSV file should contain these columns:
- `No.`: School number/ID
- `Academic Year`: Academic year (e.g., 2023-24)
- `Month od school onboarding`: Onboarding month
- `Region`: Geographic region
- `City`: City name
- `Channel`: Acquisition channel
- `POC`: Point of contact
- `School Name`: School name
- `School and Program`: Combined school and program info
- `Program`: Program type (STEM, FinLit, Comms, combinations)
- `Grade`: Grade levels served
- `Board`: Educational board (CBSE, ICSE, State Board, IB, Cambridge)
- `Fees Range`: Fee range categories
- `Strength Range`: Student strength ranges
- `Strength`: Actual student count
- `Rev/ student`: Revenue per student
- `Sum Product`: Product sum
- `Total Business`: Total business value
- `Category`: School category (Premium, Mid-tier, Budget, Elite)
- `Full year/ mid year`: Onboarding type

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run school_dashboard.py
```

### 3. Upload Your Data
- Use the file uploader in the sidebar to upload your CSV file
- If no file is uploaded, the dashboard will use sample data for demonstration

### 4. Explore the Analysis
- Navigate through the 5 tabs to explore different aspects of your data
- Use filters in the Detailed Analysis tab for focused insights
- Check individual school recommendations in the School Finder tab

## Key Insights the Dashboard Provides

### School Segmentation
The dashboard automatically segments schools into 5 categories based on:
- Student strength
- Revenue per student
- Board type
- School category
- Geographic region

### Curriculum Personalization
For each segment, the dashboard recommends:

**Elite/Premium Schools:**
- Advanced STEM with robotics, AI, coding
- Investment basics, entrepreneurship
- Public speaking, debate, international communication

**Mid-tier Schools:**
- Practical STEM with experiments, basic programming
- Personal finance, budgeting concepts
- Presentation skills, effective communication

**Budget/Local Schools:**
- Foundational STEM with hands-on activities
- Basic financial literacy, money management
- Language skills, basic communication

### Business Intelligence
- Revenue analysis by segment and region
- Program popularity across different school types
- Market penetration insights
- Growth opportunity identification

## Technical Details

### Machine Learning
- Uses K-Means clustering for school segmentation
- Features: Strength, Revenue/Student, Board, Category, Region
- Standardized features for optimal clustering
- 5 clusters optimized for education market segmentation

### Visualizations
- Interactive Plotly charts
- Responsive design with Streamlit
- Multiple chart types: scatter, bar, pie, heatmap, violin, sunburst, treemap
- Real-time filtering and updates

### Data Processing
- Automatic data validation and cleaning
- Handles missing values
- Categorical encoding for machine learning
- Sample data generation for testing

## Sample Data

If you don't have your CSV file ready, the dashboard includes realistic sample data with:
- 500 schools across India
- Realistic correlations between board types and fees
- Geographic distribution across major cities
- Program combinations and student strengths
- Revenue patterns based on school categories

## Customization

You can easily customize:
- Segment names and descriptions
- Curriculum recommendations
- Color schemes and styling
- Additional analysis features
- Export capabilities

## Support

The dashboard is designed to work with your exact column structure. If you encounter any issues:
1. Ensure your CSV has all required columns
2. Check for data type consistency
3. Verify no special characters in column names
4. Contact support for custom modifications

---

**Goal**: Help you understand school types based on board, fee structure, city, and student strength to provide personalized curriculum content for STEM, FinLit, and Communication programs.