# 📊 NLG Paper Trends Analysis UI

An interactive web application for analyzing Natural Language Generation (NLG) paper trends from major NLP conferences (ACL, EMNLP, NAACL, INLG) from 2020-2025.

## 🌐 **Live Demo**
**[🚀 Try the App Online](https://your-username-nlg-paper-analysis-app-xyz.streamlit.app)** *(Update this link after deployment)*

## 📱 **Quick Access**
- **GitHub Repository**: [View Source Code](https://github.com/your-username/nlg-paper-analysis)
- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Issues & Feedback**: [GitHub Issues](https://github.com/your-username/nlg-paper-analysis/issues)

## 🎯 Features

This application provides comprehensive analysis across 8 different sections:

### 1. General Overview
- Number and percentage of NLG papers across conferences and years
- Interactive visualizations showing trends over time
- Conference comparison statistics

### 2. NLG Papers Overview  
- Distribution of evaluation approaches (Automatic, LLM, Human, and combinations)
- Trend analysis of evaluation methods over time
- Pie charts and line graphs for evaluation type distribution

### 3. Text-Only NLG Tasks
- Filters out multimodal tasks (image captioning, visual QA, code generation, etc.)
- Identifies and analyzes top-10 text-only NLG tasks
- Evaluation approach distribution for text-only tasks

### 4. NLG Tasks Distribution (Question 1)
- Complete analysis of all NLG tasks across conferences and years
- Interactive task selection and filtering
- Heatmap visualizations showing task distribution patterns

### 5. Automatic Metrics Analysis (Question 2)
- Distribution of automatic evaluation metrics across conferences and years
- Metrics usage trends over time
- Task-specific metrics analysis with interactive heatmaps

### 6. LLM Evaluation Analysis (Question 3)
- Analysis of LLM models used as judges/evaluators
- Distribution of LLM evaluation methods and criteria
- Temporal trends in LLM evaluation approaches

### 7. Human Evaluation Analysis (Question 4)
- Human evaluation criteria distribution
- Guidelines analysis with sample extracts
- Trends in human evaluation approaches over time

### 8. Cross-Evaluation Comparison
- Papers using all three evaluation types (Automatic + LLM + Human)
- Comparison of criteria used in LLM vs Human evaluation
- Venn diagram analysis of evaluation overlaps

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (install with pip)

### Installation

1. **Navigate to the UI directory:**
   ```bash
   cd nlg-user-ui
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data file exists:**
   Make sure `data/all_papers.json` is present in the directory

4. **Run the application:**
   ```bash
   python run_app.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:8501`

## 📊 Data Structure

The application expects data in the following JSON format:

```json
[
  {
    "answer_1": {
      "answer": "Yes/No",
      "tasks": ["Task1", "Task2"],
      "datasets": ["Dataset1"], 
      "languages": ["English"],
      "models": ["Model1"],
      "outputs": "Generated outputs description"
    },
    "answer_2": {
      "answer": "Yes/No",
      "automatic_metrics": ["BLEU", "ROUGE"]
    },
    "answer_3": {
      "answer": "Yes/No", 
      "models": ["GPT-4", "Claude"],
      "methods": ["Method1"],
      "criteria": ["Fluency", "Coherence"]
    },
    "answer_4": {
      "answer": "Yes/No",
      "guideline": "Human evaluation guidelines",
      "criteria": ["Naturalness", "Informativeness"]
    },
    "paper_id": "2020.acl-main.1",
    "conference": "ACL",
    "year": 2020,
    "title": "Paper Title",
    "authors": ["Author1", "Author2"],
    "abstract": "Paper abstract...",
    "url": "https://...",
    "is_nlg": true/false
  }
]
```

## 🎨 Features & Interactions

### 🎛️ Enhanced Filtering & Controls
- **Global Paper Filtering**: Choose from 4 paper subsets:
  - All Papers (12,312 papers)
  - All NLG Papers (7,528 papers)
  - NLG Text-Only Papers (6,279 papers - excludes multimodal)
  - NLG Text-Only with All Metrics (658 papers - text-only with auto+LLM+human eval)
- **Multi-Select Conferences**: Filter by ACL, EMNLP, NAACL, INLG
- **Multi-Select Years**: Filter by years 2020-2025
- **Real-time Statistics**: Live count updates in sidebar

### 📊 Advanced Visualization Options
- **Basic Plot Types**: Bar Chart, Line Chart, Area Chart, Stacked Bar, Horizontal Bar
- **Complex Visualizations**: Heatmap, Treemap, Sunburst, Scatter Plot, Box Plot, Violin Plot
- **Multi-dimensional Analysis**: Correlation matrices, co-occurrence networks, bubble charts
- **Advanced Features**:
  - **Heatmaps**: Metrics × Tasks, Evaluation Methods × Conferences
  - **Treemaps**: Hierarchical Task → Metric relationships
  - **Sunburst Charts**: Nested categorical data visualization  
  - **Correlation Matrices**: Evaluation method relationships
  - **Network Plots**: Metric co-occurrence analysis
  - **Bubble Charts**: Multi-dimensional scatter plots with size encoding
- **Color Grouping**: Group by Conference, Year, Evaluation Type, Task, or Metric
- **Percentage Toggle**: Switch between raw counts and percentages
- **Interactive Legends**: Click to show/hide data series

### 🤖 Smart Metric Normalization
- **Automatic Merging**: Combines similar metrics (BLEU-1, BLEU-2 → BLEU-N)
- **Toggle Control**: Enable/disable normalization with one click
- **Normalization Stats**: Shows before/after metric count
- **Common Patterns**: ROUGE variants, BERT-based metrics, accuracy measures

### Interactive Elements
- **Sidebar Navigation**: Easy switching between analysis sections
- **Dynamic Filtering**: Select specific conferences, years, tasks, or metrics
- **Interactive Charts**: Hover for details, zoom, and pan
- **Expandable Sections**: Click to view detailed paper information
- **Multi-select Widgets**: Analyze multiple items simultaneously

### Visualization Types

#### 📊 Basic Charts
- **Bar Charts**: Distribution and ranking visualizations
- **Line Charts**: Temporal trend analysis  
- **Area Charts**: Filled trend visualizations
- **Pie Charts**: Proportion and percentage views

#### 🔥 Advanced Visualizations
- **Heatmaps**: 2D color-coded matrices showing relationships between two categorical variables
  - Example: Metrics (rows) × Tasks (columns) with color intensity = usage frequency
- **Treemaps**: Hierarchical data as nested rectangles, size = value
  - Example: Tasks → Metrics hierarchy with rectangle size = usage count
- **Sunburst Charts**: Multi-level pie charts showing nested categories  
  - Example: Evaluation combinations (Automatic → LLM → Human)
- **Correlation Matrices**: Heatmap showing statistical correlations (-1 to +1)
  - Example: Relationships between Automatic, LLM, Human evaluation methods
- **Bubble Charts**: Scatter plots with bubble size encoding additional dimension
  - Example: Year vs Conference with bubble size = paper count, color = metric type
- **Co-occurrence Networks**: Scatter plot showing which items appear together
  - Example: Which automatic metrics are used together in the same papers

#### 📈 Statistical Plots
- **Scatter Plots**: Relationship between two continuous variables
- **Box Plots**: Distribution showing median, quartiles, and outliers
- **Violin Plots**: Distribution density with box plot overlay

#### 📋 Data Tables
- **Interactive Tables**: Sortable, filterable detailed statistics
- **Pivot Tables**: Cross-tabulated data with totals

### Key Metrics Displayed
- Total papers and NLG paper counts
- Evaluation method usage statistics
- Task popularity rankings
- Conference and year breakdowns
- Percentage distributions

## 🛠️ Technical Details

### Built With
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Architecture
- Modular design with separate functions for each analysis section
- Efficient data caching using `@st.cache_data`
- Clean separation of data processing and visualization logic

## 📈 Usage Examples

### Example 1: Analyzing Evaluation Trends
1. Go to "NLG Papers Overview"
2. View the trend line showing how evaluation approaches evolved
3. Compare automatic vs. LLM vs. human evaluation adoption rates

### Example 2: Finding Popular Tasks
1. Navigate to "Text-Only NLG Tasks"
2. See the top-10 most common text-only NLG tasks
3. Analyze evaluation patterns for these popular tasks

### Example 3: Advanced Metric Analysis
1. Visit "Automatic Metrics Analysis (Q2)"
2. Enable "Normalize Metric Names" to merge similar metrics
3. Select subset "NLG Text-Only Papers" for focused analysis
4. Choose "Line Chart" and group by "Conference"
5. Toggle to "Show Percentages" for relative comparison
6. Select specific metrics to track trends over time

### Example 4: Complex Heatmap Analysis
1. Go to "Automatic Metrics Analysis (Q2)"
2. Enable "Normalize Metric Names" and "Advanced Analysis"
3. View the **Metrics × Tasks Heatmap** showing which metrics are used for which tasks
4. Explore the **Treemap** showing hierarchical Task → Metric relationships
5. Analyze the **Metric Co-occurrence Network** to see which metrics are used together

### Example 5: Evaluation Methods Correlation
1. Navigate to "NLG Papers Overview"
2. Enable "Evaluation Matrix" checkbox
3. View the **Correlation Matrix** showing relationships between Automatic, LLM, and Human evaluation
4. Analyze the **Conference-wise Adoption Heatmap**
5. Explore the **Sunburst Chart** showing evaluation method combinations

### Example 6: Multi-Dimensional Filtering with Advanced Plots
1. Set Paper Subset to "NLG Text-Only with All Metrics" (658 papers)
2. Select conferences: ["ACL", "EMNLP"] 
3. Select years: [2022, 2023, 2024]
4. Use "Heatmap" plot type to see Conference × Year relationships
5. Switch to "Bubble Chart" to explore multi-dimensional patterns

## 🔧 Troubleshooting

### Common Issues

**Data file not found:**
```
❌ Error: data/all_papers.json not found!
```
- Ensure the data file is in the `data/` directory
- Check that the file path is correct

**Import errors:**
```
ModuleNotFoundError: No module named 'streamlit'
```
- Install required packages: `pip install -r requirements.txt`

**Empty visualizations:**
- Check that your data file contains valid JSON
- Verify that papers have the expected answer fields

### Performance Tips
- The app caches data automatically for faster loading
- For large datasets, some visualizations might take time to render
- Use filters to focus on specific subsets for faster interaction

## 📚 Understanding the Analysis

### Research Questions Addressed
1. **Does the paper address NLG tasks?** (answer_1)
2. **Does the paper use automatic metrics?** (answer_2)  
3. **Does the paper use LLMs as judges?** (answer_3)
4. **Does the paper conduct human evaluations?** (answer_4)

### Key Insights Available
- Evolution of evaluation practices in NLG research
- Most popular NLG tasks and their evaluation patterns
- Adoption trends of different evaluation methodologies
- Comparison between automatic, LLM, and human evaluation approaches

## 🤝 Contributing

To add new analysis features:

1. Create a new function following the pattern: `show_[analysis_name](data)`
2. Add the option to the sidebar selectbox in `main()`
3. Implement the analysis logic and visualizations
4. Test with the sample data

## 📄 License

This project is part of the NLG evaluation research initiative. Please refer to the main project license for usage terms.