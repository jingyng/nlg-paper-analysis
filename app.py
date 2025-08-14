import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict, Counter
import re

st.set_page_config(
    page_title="NLG Paper Trends Analysis",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    with open('data/all_papers.json', 'r') as f:
        data = json.load(f)
    return data

def normalize_metric_name(metric):
    """Normalize metric names to merge similar/overlapping metrics"""
    metric = metric.strip().lower()
    
    # BLEU variations
    if 'bleu' in metric:
        if any(x in metric for x in ['1', '2', '3', '4']):
            return 'BLEU-N'
        elif 'sentence' in metric or 'sbleu' in metric:
            return 'Sentence-BLEU'
        elif 'self' in metric:
            return 'Self-BLEU'
        else:
            return 'BLEU'
    
    # ROUGE variations
    if 'rouge' in metric:
        if 'rouge-1' in metric:
            return 'ROUGE-1'
        elif 'rouge-2' in metric:
            return 'ROUGE-2'
        elif 'rouge-l' in metric:
            return 'ROUGE-L'
        elif 'rouge-w' in metric:
            return 'ROUGE-W'
        else:
            return 'ROUGE'
    
    # METEOR variations
    if 'meteor' in metric:
        return 'METEOR'
    
    # CIDEr variations
    if 'cider' in metric:
        return 'CIDEr'
    
    # BERT-based metrics
    if 'bertscore' in metric or 'bert-score' in metric or 'bert score' in metric:
        return 'BERTScore'
    if 'bleurt' in metric:
        return 'BLEURT'
    
    # Perplexity variations
    if 'perplexity' in metric or 'ppl' in metric:
        return 'Perplexity'
    
    # Exact match variations
    if 'exact match' in metric or 'exact_match' in metric or 'em' in metric:
        return 'Exact Match'
    
    # F1 score variations
    if 'f1' in metric or 'f-1' in metric:
        return 'F1 Score'
    
    # Accuracy variations
    if 'accuracy' in metric or 'acc' in metric:
        return 'Accuracy'
    
    # Diversity metrics
    if 'distinct' in metric or 'diversity' in metric:
        return 'Diversity'
    
    # Semantic similarity
    if 'semantic' in metric and 'similarity' in metric:
        return 'Semantic Similarity'
    
    # Error rates
    if 'error rate' in metric or 'err' in metric:
        return 'Error Rate'
    
    # Return original if no normalization rule applies
    return metric.title()

def get_paper_subset(data, subset_type, conferences=None, years=None):
    """Filter papers based on subset type and additional filters"""
    filtered_data = data.copy()
    
    # Apply conference filter
    if conferences:
        filtered_data = [p for p in filtered_data if p['conference'] in conferences]
    
    # Apply year filter
    if years:
        filtered_data = [p for p in filtered_data if p['year'] in years]
    
    # Apply subset filter
    if subset_type == "All Papers":
        return filtered_data
    elif subset_type == "All NLG Papers":
        return [p for p in filtered_data if p['answer_1']['answer'] == 'Yes']
    elif subset_type == "NLG Text-Only Papers":
        nlg_papers = [p for p in filtered_data if p['answer_1']['answer'] == 'Yes']
        text_only_papers = []
        for paper in nlg_papers:
            tasks = paper['answer_1'].get('tasks', [])
            if tasks and is_text_only_task(tasks):
                text_only_papers.append(paper)
        return text_only_papers
    elif subset_type == "NLG Text-Only with All Metrics":
        nlg_papers = [p for p in filtered_data if p['answer_1']['answer'] == 'Yes']
        text_only_papers = []
        for paper in nlg_papers:
            tasks = paper['answer_1'].get('tasks', [])
            if tasks and is_text_only_task(tasks):
                # Check if has all three evaluation types
                if (paper['answer_2']['answer'] == 'Yes' and 
                    paper['answer_3']['answer'] == 'Yes' and 
                    paper['answer_4']['answer'] == 'Yes'):
                    text_only_papers.append(paper)
        return text_only_papers
    else:
        return filtered_data

def clean_task_name(task):
    if task.startswith("Other: ") or task.startswith("Other:"):
        return re.sub(r"^Other:\s*", "", task)
    return task

def get_evaluation_type(paper):
    automatic = paper['answer_2']['answer'] == 'Yes'
    llm = paper['answer_3']['answer'] == 'Yes'
    human = paper['answer_4']['answer'] == 'Yes'
    
    if automatic and llm and human:
        return 'All Three'
    elif automatic and llm:
        return 'Automatic + LLM'
    elif automatic and human:
        return 'Automatic + Human'
    elif llm and human:
        return 'LLM + Human'
    elif automatic:
        return 'Automatic Only'
    elif llm:
        return 'LLM Only'
    elif human:
        return 'Human Only'
    else:
        return 'None'

def is_text_only_task(tasks):
    multimodal_keywords = [
        'image', 'visual', 'video', 'audio', 'speech', 'code', 'captioning',
        'vision', 'multimodal', 'cross-modal', 'picture', 'photo', 'graphic'
    ]
    for task in tasks:
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in multimodal_keywords):
            return False
    return True

def calculate_conference_year_percentages(data, filtered_data):
    """Calculate percentages relative to total papers per conference-year combination"""
    # Get total papers per conference-year
    total_counts = defaultdict(int)
    for paper in data:  # Use original unfiltered data for totals
        key = (paper['conference'], paper['year'])
        total_counts[key] += 1
    
    # Get filtered counts per conference-year
    filtered_counts = defaultdict(int)
    for paper in filtered_data:
        key = (paper['conference'], paper['year'])
        filtered_counts[key] += 1
    
    # Calculate percentages
    percentage_data = []
    for (conference, year), filtered_count in filtered_counts.items():
        total_count = total_counts.get((conference, year), 0)
        percentage = (filtered_count / total_count * 100) if total_count > 0 else 0
        percentage_data.append({
            'Conference': conference,
            'Year': year,
            'Count': filtered_count,
            'Total': total_count,
            'Percentage': percentage
        })
    
    return pd.DataFrame(percentage_data)

def create_plot(df, plot_type, x_col, y_col, color_col=None, title="", use_percentage=False, height=400, conference_year_totals=None):
    """Create different types of plots based on user selection"""
    
    if use_percentage:
        # Use existing percentage column if available, otherwise calculate
        if 'Percentage' in df.columns:
            y_col = 'Percentage'
        elif 'Count' in df.columns:
            # Convert counts to percentages with context-aware calculation
            df_pct = df.copy()
            
            # For conference-year data with proper totals available
            if 'Total' in df.columns:
                # Use pre-calculated percentages relative to conference-year totals
                df_pct['Percentage'] = df_pct.apply(
                    lambda row: (row['Count'] / row['Total'] * 100) if row['Total'] > 0 else 0, axis=1
                )
                y_col = 'Percentage'
            elif color_col and plot_type in ["Bar Chart", "Stacked Bar", "Line Chart"]:
                # For grouped charts, calculate percentages within each group
                df_pct['Percentage'] = df_pct.groupby(color_col)['Count'].transform(lambda x: x / x.sum() * 100)
                y_col = 'Percentage'
            else:
                # Default: global percentage calculation
                df_pct['Percentage'] = df_pct['Count'] / df_pct['Count'].sum() * 100
                y_col = 'Percentage'
            df = df_pct
    
    if plot_type == "Bar Chart":
        if color_col:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, barmode='group')
        else:
            fig = px.bar(df, x=x_col, y=y_col, title=title)
    
    elif plot_type == "Line Chart":
        if color_col:
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title, markers=True)
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
    
    elif plot_type == "Area Chart":
        if color_col:
            fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.area(df, x=x_col, y=y_col, title=title)
    
    elif plot_type == "Stacked Bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, barmode='stack')
    
    elif plot_type == "Horizontal Bar":
        if color_col:
            fig = px.bar(df, x=y_col, y=x_col, color=color_col, title=title, orientation='h', barmode='group')
        else:
            fig = px.bar(df, x=y_col, y=x_col, title=title, orientation='h')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    elif plot_type == "Heatmap":
        # Create pivot table for heatmap
        if color_col and len(df.columns) >= 3:
            pivot_df = df.pivot_table(values=y_col, index=x_col, columns=color_col, fill_value=0, aggfunc='sum')
            fig = px.imshow(pivot_df.values, 
                          x=pivot_df.columns, 
                          y=pivot_df.index,
                          aspect='auto',
                          title=title,
                          color_continuous_scale='Blues',
                          labels={'color': y_col})
            fig.update_layout(xaxis_title=color_col, yaxis_title=x_col)
        else:
            # Simple heatmap with just x and y
            fig = px.density_heatmap(df, x=x_col, y=y_col, title=title)
    
    elif plot_type == "Treemap":
        if color_col:
            fig = px.treemap(df, path=[color_col, x_col], values=y_col, title=title)
        else:
            fig = px.treemap(df, path=[x_col], values=y_col, title=title)
    
    elif plot_type == "Sunburst":
        if color_col:
            fig = px.sunburst(df, path=[color_col, x_col], values=y_col, title=title)
        else:
            fig = px.sunburst(df, path=[x_col], values=y_col, title=title)
    
    elif plot_type == "Scatter Plot":
        if color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title, size=y_col)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=title, size=y_col)
    
    elif plot_type == "Box Plot":
        if color_col:
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.box(df, x=x_col, y=y_col, title=title)
    
    elif plot_type == "Violin Plot":
        if color_col:
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.violin(df, x=x_col, y=y_col, title=title)
    
    else:  # Default to bar chart
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
    
    fig.update_layout(height=height)
    return fig

def create_advanced_heatmap(df, x_col, y_col, value_col, title="", height=500):
    """Create advanced heatmap with better formatting"""
    
    # Create pivot table
    pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, fill_value=0, aggfunc='sum')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='Blues',
        text=pivot_df.values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{value_col}: %{{z}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=height,
        font=dict(size=10)
    )
    
    return fig

def create_correlation_matrix(df, title="Correlation Matrix"):
    """Create correlation matrix heatmap"""
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig

def create_multi_dimensional_plot(df, metrics_col, tasks_col, conferences_col, years_col, title=""):
    """Create complex multi-dimensional visualization"""
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Metrics by Task', 'Trends by Conference', 
                       'Distribution Heatmap', 'Summary Statistics'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "pie"}]]
    )
    
    # Plot 1: Metrics by Task (Bar)
    metrics_tasks = df.groupby([tasks_col, metrics_col]).size().reset_index(name='count')
    top_tasks = metrics_tasks.groupby(tasks_col)['count'].sum().nlargest(10).index
    filtered_data = metrics_tasks[metrics_tasks[tasks_col].isin(top_tasks)]
    
    for task in top_tasks[:5]:  # Show top 5 tasks
        task_data = filtered_data[filtered_data[tasks_col] == task]
        fig.add_trace(
            go.Bar(x=task_data[metrics_col], y=task_data['count'], 
                  name=task, showlegend=True),
            row=1, col=1
        )
    
    # Plot 2: Trends by Conference (Line)
    yearly_trends = df.groupby([years_col, conferences_col]).size().reset_index(name='count')
    for conf in df[conferences_col].unique():
        conf_data = yearly_trends[yearly_trends[conferences_col] == conf]
        fig.add_trace(
            go.Scatter(x=conf_data[years_col], y=conf_data['count'],
                      mode='lines+markers', name=f'{conf}', showlegend=True),
            row=1, col=2
        )
    
    fig.update_layout(height=800, title_text=title)
    return fig

def main():
    st.title("📊 NLG Paper Trends Analysis (2020-2025)")
    st.markdown("Analysis of Natural Language Generation papers from major NLP conferences")
    
    # Load data
    try:
        data = load_data()
        st.success(f"Loaded {len(data)} papers from ACL, EMNLP, NAACL, and INLG conferences")
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/all_papers.json' exists.")
        return
    
    # Global filters in sidebar
    st.sidebar.title("🎛️ Global Filters")
    
    # Paper subset selection
    paper_subset = st.sidebar.selectbox(
        "Paper Subset:",
        ["All Papers", "All NLG Papers", "NLG Text-Only Papers", "NLG Text-Only with All Metrics"],
        index=1,
        help="Choose which papers to analyze"
    )
    
    # Conference selection
    all_conferences = sorted(set(p['conference'] for p in data))
    selected_conferences = st.sidebar.multiselect(
        "Conferences:",
        options=all_conferences,
        default=all_conferences,
        help="Select conferences to include in analysis"
    )
    
    # Year selection
    all_years = sorted(set(p['year'] for p in data))
    selected_years = st.sidebar.multiselect(
        "Years:",
        options=all_years,
        default=all_years,
        help="Select years to include in analysis"
    )
    
    # Apply filters to get working dataset
    if selected_conferences and selected_years:
        filtered_data = get_paper_subset(data, paper_subset, selected_conferences, selected_years)
    else:
        filtered_data = []
        st.warning("Please select at least one conference and one year.")
    
    # Display filtered data statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Current Selection:**")
    st.sidebar.metric("Papers in Selection", f"{len(filtered_data):,}")
    if paper_subset != "All Papers" and filtered_data:
        original_count = len(get_paper_subset(data, paper_subset))
        st.sidebar.metric("Total in Subset", f"{original_count:,}")
    
    # Sidebar navigation
    st.sidebar.markdown("---")
    st.sidebar.title("📋 Analysis Sections")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis:",
        [
            "General Overview",
            "NLG Papers Overview", 
            "Text-Only NLG Tasks",
            "NLG Tasks Distribution (Q1)",
            "Automatic Metrics Analysis (Q2)",
            "LLM Evaluation Analysis (Q3)", 
            "Human Evaluation Analysis (Q4)",
            "Cross-Evaluation Comparison"
        ]
    )
    
    # Show analysis based on selection
    if not filtered_data:
        st.warning("No papers match the current filter selection.")
        return
        
    if analysis_type == "General Overview":
        show_general_overview(filtered_data, paper_subset, selected_conferences, selected_years)
    elif analysis_type == "NLG Papers Overview":
        show_nlg_overview(filtered_data, paper_subset)
    elif analysis_type == "Text-Only NLG Tasks":
        show_text_only_analysis(filtered_data, paper_subset)
    elif analysis_type == "NLG Tasks Distribution (Q1)":
        show_tasks_distribution(filtered_data, paper_subset)
    elif analysis_type == "Automatic Metrics Analysis (Q2)":
        show_automatic_metrics_analysis(filtered_data, paper_subset)
    elif analysis_type == "LLM Evaluation Analysis (Q3)":
        show_llm_evaluation_analysis(filtered_data, paper_subset)
    elif analysis_type == "Human Evaluation Analysis (Q4)":
        show_human_evaluation_analysis(filtered_data, paper_subset)
    elif analysis_type == "Cross-Evaluation Comparison":
        show_cross_evaluation_analysis(filtered_data, paper_subset)

def show_general_overview(data, paper_subset, selected_conferences=None, selected_years=None):
    st.header("🌟 General Overview")
    st.markdown(f"Analysis of **{paper_subset}** across conferences and years")
    
    # Plot controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", 
                                ["Bar Chart", "Line Chart", "Area Chart", "Stacked Bar", "Heatmap", "Treemap", "Sunburst"], 
                                key="overview_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="overview_pct")
    with col3:
        group_by = st.selectbox("Color By:", ["Conference", "Year"], key="overview_color")
    
    # Create summary statistics
    total_papers = len(data)
    
    # Get appropriate counts based on subset
    if paper_subset == "All Papers":
        nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes']
        total_nlg = len(nlg_papers)
        main_metric = "NLG Papers"
    else:
        total_nlg = total_papers  # Already filtered
        main_metric = "Selected Papers"
    
    # Statistics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total in Selection", f"{total_papers:,}")
    with col2:
        if paper_subset == "All Papers":
            st.metric("NLG Papers", f"{total_nlg:,}")
        else:
            st.metric(main_metric, f"{total_nlg:,}")
    with col3:
        if paper_subset == "All Papers":
            pct = (total_nlg/total_papers*100) if total_papers > 0 else 0
            st.metric("NLG Percentage", f"{pct:.1f}%")
        else:
            # Show evaluation stats for filtered subsets
            if total_papers > 0:
                auto_count = len([p for p in data if p['answer_2']['answer'] == 'Yes'])
                st.metric("With Auto Eval", f"{auto_count:,} ({auto_count/total_papers*100:.1f}%)")
    
    # Conference and year breakdown
    conference_year_stats = defaultdict(lambda: {'total': 0})
    
    for paper in data:
        key = (paper['conference'], paper['year'])
        conference_year_stats[key]['total'] += 1
    
    # Prepare data for visualization
    viz_data = []
    for (conference, year), stats in conference_year_stats.items():
        viz_data.append({
            'Conference': conference,
            'Year': year,
            'Count': stats['total']
        })
    
    if viz_data:
        df = pd.DataFrame(viz_data)
        
        # For percentage calculations, we need context-aware totals
        if show_percentage:
            # Get the original unfiltered data
            original_data = load_data()
            
            # Filter original data only by conference and year (not by paper subset)
            baseline_data = []
            for paper in original_data:
                if (not selected_conferences or paper['conference'] in selected_conferences) and \
                   (not selected_years or paper['year'] in selected_years):
                    baseline_data.append(paper)
            
            # Calculate conference-year relative percentages
            # baseline_data = total papers per conference-year (denominator)
            # data = papers matching paper subset + conference + year filters (numerator) 
            percentage_df = calculate_conference_year_percentages(baseline_data, data)
            
            # Merge with viz_data to get the correct structure
            df = df.merge(
                percentage_df[['Conference', 'Year', 'Total', 'Percentage']], 
                on=['Conference', 'Year'], 
                how='left'
            ).fillna({'Total': 0, 'Percentage': 0})
        
        # Create the main visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"{main_metric} Distribution")
            
            # Determine x and color columns based on user selection
            if group_by == "Conference":
                x_col, color_col = 'Year', 'Conference'
            else:
                x_col, color_col = 'Conference', 'Year'
                # Convert year to string for better color mapping
                df['Year'] = df['Year'].astype(str)
            
            fig = create_plot(df, plot_type, x_col, 'Count', color_col, 
                            f"{main_metric} by {x_col} and {color_col}", 
                            show_percentage, 500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Summary Statistics")
            
            if show_percentage and 'Percentage' in df.columns:
                # Show percentages when in percentage mode
                st.write("**By Conference (avg %):**")
                conf_percentages = df.groupby('Conference')['Percentage'].mean().sort_values(ascending=False)
                for conf, pct in conf_percentages.items():
                    st.write(f"• {conf}: {pct:.1f}%")
                
                st.write("**By Year (avg %):**")
                year_percentages = df.groupby('Year')['Percentage'].mean().sort_index()
                for year, pct in year_percentages.items():
                    st.write(f"• {year}: {pct:.1f}%")
            else:
                # Show raw counts
                conf_totals = df.groupby('Conference')['Count'].sum().sort_values(ascending=False)
                st.write("**By Conference:**")
                for conf, count in conf_totals.items():
                    st.write(f"• {conf}: {count:,}")
                
                # Year totals
                st.write("**By Year:**")
                year_totals = df.groupby('Year')['Count'].sum().sort_index()
                for year, count in year_totals.items():
                    st.write(f"• {year}: {count:,}")
        
        # Detailed table
        st.subheader("Detailed Breakdown")
        
        if show_percentage and 'Percentage' in df.columns:
            # Show percentage table with additional context
            st.write("*Percentages are relative to total papers per conference-year*")
            display_df = df[['Conference', 'Year', 'Count', 'Total', 'Percentage']].copy()
            display_df['Percentage'] = display_df['Percentage'].round(1)
            st.dataframe(display_df.sort_values(['Conference', 'Year']), use_container_width=True)
        else:
            # Show count pivot table
            pivot_df = df.pivot_table(values='Count', index='Conference', columns='Year', fill_value=0)
            
            # Add totals
            pivot_df['Total'] = pivot_df.sum(axis=1)
            pivot_df.loc['Total'] = pivot_df.sum()
            
            st.dataframe(pivot_df, use_container_width=True)
    else:
        st.warning("No data available for the current selection.")

def show_nlg_overview(data, paper_subset):
    st.header("📖 NLG Papers Overview")
    st.markdown(f"Distribution of evaluation approaches in **{paper_subset}**")
    
    # Filter to NLG papers if not already filtered
    if paper_subset == "All Papers":
        nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes']
    else:
        nlg_papers = data  # Already NLG papers
    
    if not nlg_papers:
        st.warning("No NLG papers in current selection.")
        return
        
    # Plot controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", 
                                ["Line Chart", "Bar Chart", "Area Chart", "Heatmap", "Sunburst", "Treemap"], 
                                key="nlg_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="nlg_pct")
    with col3:
        group_by = st.selectbox("Color By:", ["Evaluation Type", "Conference"], key="nlg_color")
    with col4:
        show_eval_matrix = st.checkbox("Evaluation Matrix", value=False, key="nlg_matrix",
                                      help="Show evaluation methods correlation matrix")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total NLG Papers", f"{len(nlg_papers):,}")
    with col2:
        automatic_count = len([p for p in nlg_papers if p['answer_2']['answer'] == 'Yes'])
        st.metric("With Automatic Eval", f"{automatic_count:,}")
    with col3:
        llm_count = len([p for p in nlg_papers if p['answer_3']['answer'] == 'Yes'])
        st.metric("With LLM Eval", f"{llm_count:,}")
    with col4:
        human_count = len([p for p in nlg_papers if p['answer_4']['answer'] == 'Yes'])
        st.metric("With Human Eval", f"{human_count:,}")
    
    # Evaluation approaches distribution over time
    eval_data = []
    for paper in nlg_papers:
        eval_type = get_evaluation_type(paper)
        eval_data.append({
            'Year': paper['year'],
            'Conference': paper['conference'],
            'Evaluation Type': eval_type
        })
    
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        
        # Aggregate data
        if group_by == "Evaluation Type":
            agg_df = eval_df.groupby(['Year', 'Evaluation Type']).size().reset_index(name='Count')
            x_col, color_col = 'Year', 'Evaluation Type'
        else:
            agg_df = eval_df.groupby(['Year', 'Conference']).size().reset_index(name='Count')
            x_col, color_col = 'Year', 'Conference'
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Evaluation Approaches Over Time")
            fig = create_plot(agg_df, plot_type, x_col, 'Count', color_col,
                            f"Trends in {color_col}", show_percentage, 500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Overall Distribution")
            
            # Overall evaluation type distribution
            overall_eval_counts = eval_df['Evaluation Type'].value_counts()
            
            eval_pie_df = pd.DataFrame({
                'Type': overall_eval_counts.index,
                'Count': overall_eval_counts.values
            })
            
            if not eval_pie_df.empty:
                fig = px.pie(eval_pie_df, values='Count', names='Type',
                            title="Distribution of Evaluation Types")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Evaluation Methods Correlation Matrix
        if show_eval_matrix:
            st.subheader("📊 Evaluation Methods Analysis")
            
            # Create evaluation matrix
            eval_matrix_data = []
            for paper in nlg_papers:
                eval_matrix_data.append({
                    'Paper': f"{paper['conference']}_{paper['year']}",
                    'Automatic': 1 if paper['answer_2']['answer'] == 'Yes' else 0,
                    'LLM': 1 if paper['answer_3']['answer'] == 'Yes' else 0,
                    'Human': 1 if paper['answer_4']['answer'] == 'Yes' else 0,
                    'Conference': paper['conference'],
                    'Year': paper['year']
                })
            
            eval_matrix_df = pd.DataFrame(eval_matrix_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation matrix of evaluation methods
                eval_methods = ['Automatic', 'LLM', 'Human']
                corr_data = eval_matrix_df[eval_methods].corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=eval_methods,
                    y=eval_methods,
                    colorscale='RdYlBu',
                    zmid=0,
                    text=np.round(corr_data.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
                ))
                
                fig_corr.update_layout(
                    title="Evaluation Methods Correlation",
                    height=400
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # Conference-wise evaluation adoption heatmap
                conf_eval_data = eval_matrix_df.groupby('Conference')[eval_methods].mean().reset_index()
                conf_eval_melted = pd.melt(conf_eval_data, id_vars=['Conference'], 
                                         value_vars=eval_methods, 
                                         var_name='Evaluation_Method', value_name='Adoption_Rate')
                
                fig_conf_heatmap = create_advanced_heatmap(
                    conf_eval_melted, 'Evaluation_Method', 'Conference', 'Adoption_Rate',
                    "Evaluation Method Adoption by Conference", 400
                )
                st.plotly_chart(fig_conf_heatmap, use_container_width=True)
            
            # Temporal evolution of evaluation methods
            st.subheader("📈 Evaluation Methods Evolution")
            
            yearly_eval = eval_matrix_df.groupby('Year')[eval_methods].mean().reset_index()
            yearly_eval_melted = pd.melt(yearly_eval, id_vars=['Year'], 
                                       value_vars=eval_methods,
                                       var_name='Evaluation_Method', value_name='Adoption_Rate')
            
            fig_evolution = px.line(yearly_eval_melted, x='Year', y='Adoption_Rate', 
                                  color='Evaluation_Method', markers=True,
                                  title="Evolution of Evaluation Method Adoption Rates",
                                  labels={'Adoption_Rate': 'Adoption Rate (0-1)'})
            fig_evolution.update_layout(height=400)
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Evaluation combinations sunburst
            eval_combinations = []
            for _, row in eval_matrix_df.iterrows():
                combination = []
                if row['Automatic']: combination.append('Automatic')
                if row['LLM']: combination.append('LLM') 
                if row['Human']: combination.append('Human')
                
                if combination:
                    eval_combinations.append(' + '.join(combination))
                else:
                    eval_combinations.append('None')
            
            combo_counts = Counter(eval_combinations)
            combo_df = pd.DataFrame([
                {'Combination': combo, 'Count': count}
                for combo, count in combo_counts.items()
            ])
            
            if len(combo_df) > 0:
                fig_combo = px.sunburst(
                    combo_df,
                    path=['Combination'],
                    values='Count',
                    title="Evaluation Method Combinations"
                )
                fig_combo.update_layout(height=400)
                st.plotly_chart(fig_combo, use_container_width=True)

def show_automatic_metrics_analysis(data, paper_subset):
    st.header("🤖 Automatic Metrics Analysis")
    st.markdown(f"Distribution of automatic evaluation metrics in **{paper_subset}**")
    
    # Plot controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", 
                                ["Horizontal Bar", "Bar Chart", "Line Chart", "Heatmap", "Treemap", "Scatter Plot"], 
                                key="metrics_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="metrics_pct")
    with col3:
        normalize_metrics = st.checkbox("Normalize Metric Names", value=True, key="metrics_normalize", 
                                      help="Merge similar metrics (e.g., BLEU-1, BLEU-2 → BLEU-N)")
    with col4:
        analysis_focus = st.selectbox("Analysis Focus:", 
                                    ["Metrics", "Task-Metrics"], 
                                    key="metrics_focus")
    with col5:
        show_advanced = st.checkbox("Advanced Analysis", value=False, key="metrics_advanced",
                                   help="Show metrics-tasks relationships and multi-dimensional plots")
    
    # Filter papers with automatic metrics
    auto_papers = [p for p in data if p['answer_2']['answer'] == 'Yes']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers in Subset", f"{len(data):,}")
    with col2:
        st.metric("Papers with Auto Metrics", f"{len(auto_papers):,}")
    with col3:
        percentage = (len(auto_papers) / len(data) * 100) if data else 0
        st.metric("Percentage with Auto Metrics", f"{percentage:.1f}%")
    
    if not auto_papers:
        st.warning("No papers with automatic metrics in current selection.")
        return
    
    # Collect all automatic metrics with normalization
    all_metrics = []
    for paper in auto_papers:
        metrics = paper['answer_2'].get('automatic_metrics', [])
        for metric in metrics:
            normalized_metric = normalize_metric_name(metric) if normalize_metrics else metric
            all_metrics.append({
                'metric': normalized_metric,
                'original_metric': metric,
                'conference': paper['conference'],
                'year': paper['year'],
                'tasks': [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
            })
    
    if all_metrics:
        # Analysis based on focus
        if analysis_focus == "Metrics":
            show_metrics_basic_analysis(all_metrics, plot_type, show_percentage, normalize_metrics, auto_papers)
        elif analysis_focus == "Task-Metrics":
            show_task_metrics_analysis(all_metrics, plot_type, show_percentage, show_advanced)
    else:
        st.warning("No automatic metrics data available in current selection.")

def show_metrics_basic_analysis(all_metrics, plot_type, show_percentage, normalize_metrics, auto_papers):
    st.subheader("📊 Automatic Metrics Overview")
    
    # Metrics popularity
    metric_counts = Counter([item['metric'] for item in all_metrics])
    top_metrics = metric_counts.most_common(20)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Most Used Automatic Metrics")
        
        if top_metrics:
            metrics_df = pd.DataFrame([
                {'Metric': metric, 'Count': count}
                for metric, count in top_metrics
            ])
            
            fig = create_plot(metrics_df, plot_type, 'Metric', 'Count', None,
                            "Automatic Metrics Usage", show_percentage, 600)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Metrics Statistics")
        st.metric("Total Metric Instances", f"{len(all_metrics):,}")
        st.metric("Unique Metrics", f"{len(metric_counts):,}")
        avg_metrics = len(all_metrics) / len(auto_papers) if auto_papers else 0
        st.metric("Avg Metrics/Paper", f"{avg_metrics:.1f}")
        
        st.subheader("Top Metrics")
        for i, (metric, count) in enumerate(top_metrics[:10], 1):
            st.write(f"**{i}. {metric}**: {count}")
        
        # Show normalization info if enabled
        if normalize_metrics:
            original_count = len(set(item['original_metric'] for item in all_metrics))
            st.info(f"📊 Normalized from {original_count} to {len(metric_counts)} unique metrics")
    
    # Metrics trends over time
    st.subheader("Metrics Usage Trends")
    
    # Select metrics for trend analysis
    selected_metrics = st.multiselect(
        "Select metrics to analyze trends:",
        options=[metric for metric, _ in top_metrics],
        default=[metric for metric, _ in top_metrics[:5]],
        key="selected_metrics"
    )
    
    if selected_metrics:
        trend_data = []
        for item in all_metrics:
            if item['metric'] in selected_metrics:
                trend_data.append({
                    'Year': item['year'],
                    'Conference': item['conference'],
                    'Metric': item['metric']
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            yearly_counts = trend_df.groupby(['Year', 'Metric']).size().reset_index(name='Count')
            
            fig = create_plot(yearly_counts, "Line Chart", 'Year', 'Count', 'Metric',
                            "Automatic Metrics Usage Over Time", False, 400)
            st.plotly_chart(fig, use_container_width=True)

def show_task_metrics_analysis(all_metrics, plot_type, show_percentage, show_advanced):
    st.subheader("🎯📊 Task-Metrics Combined Analysis")
    st.markdown("Analyzing the relationship between NLG tasks and automatic evaluation metrics")
    
    # Extract task-metrics pairs
    task_metric_pairs = []
    for item in all_metrics:
        tasks = item['tasks']
        metric = item['metric']
        
        for task in tasks[:3]:  # Limit tasks per paper
            if task and metric:
                task_metric_pairs.append({
                    'Task': task,
                    'Metric': metric,
                    'Conference': item['conference'],
                    'Year': item['year']
                })
    
    if not task_metric_pairs:
        st.warning("No task-metric pairs found.")
        return
    
    tm_df = pd.DataFrame(task_metric_pairs)
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Tasks", tm_df['Task'].nunique())
    with col2:
        st.metric("Unique Metrics", tm_df['Metric'].nunique())
    with col3:
        st.metric("Task-Metric Pairs", len(task_metric_pairs))
    
    # Task-Metrics Relationship Analysis
    st.subheader("📊 Automatic Metrics Usage by Task")
    
    # Get top tasks and metrics for cleaner visualization
    top_tasks = tm_df['Task'].value_counts().head(10).index
    top_metrics = tm_df['Metric'].value_counts().head(15).index
    
    # Filter data for better visualization
    filtered_tm = tm_df[
        (tm_df['Task'].isin(top_tasks)) & 
        (tm_df['Metric'].isin(top_metrics))
    ]
    
    if not filtered_tm.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if plot_type == "Heatmap":
                # Create metrics-task heatmap (similar to human evaluation style)
                tm_counts = filtered_tm.groupby(['Metric', 'Task']).size().reset_index(name='Count')
                heatmap_pivot = tm_counts.pivot(index='Metric', columns='Task', values='Count').fillna(0)
                
                fig_heatmap = px.imshow(heatmap_pivot,
                                      labels=dict(x="Tasks", y="Automatic Metrics", color="Usage Count"),
                                      title="Automatic Metrics Usage by NLG Task",
                                      color_continuous_scale="Blues")
                fig_heatmap.update_layout(height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                # Use tasks as color indicators for other plot types
                metrics_counts = filtered_tm.groupby(['Metric', 'Task']).size().reset_index(name='Count')
                
                fig = create_plot(metrics_counts, plot_type, 'Metric', 'Count', 'Task',
                                "Automatic Metrics Usage with Task Indicators", show_percentage, 500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Metrics-Task Summary")
            
            # Show most used metrics
            metrics_totals = filtered_tm.groupby('Metric').size().sort_values(ascending=False)
            st.write("**Most Used Metrics:**")
            for i, (metric, count) in enumerate(metrics_totals.head(8).items(), 1):
                st.write(f"{i}. **{metric}**: {count}")
    
    # Additional Analysis: Metrics Distribution Across Tasks
    st.subheader("🎯 Metrics Distribution Analysis")
    
    if not filtered_tm.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Evaluated Tasks")
            
            # Show tasks with metrics usage, using tasks as the main dimension
            task_metrics_summary = filtered_tm.groupby(['Task', 'Metric']).size().reset_index(name='Count')
            task_totals = task_metrics_summary.groupby('Task')['Count'].sum().sort_values(ascending=False)
            
            # Create bar chart with task totals
            task_totals_df = pd.DataFrame({
                'Task': task_totals.index,
                'Total_Metrics_Usage': task_totals.values
            })
            
            fig_tasks = px.bar(task_totals_df, x='Task', y='Total_Metrics_Usage',
                             title="Tasks by Automatic Metrics Usage",
                             text='Total_Metrics_Usage')
            fig_tasks.update_layout(xaxis_tickangle=-45, height=400)
            fig_tasks.update_traces(textposition='outside')
            st.plotly_chart(fig_tasks, use_container_width=True)
        
        with col2:
            st.subheader("Task-Specific Metrics")
            
            # Interactive task selector
            selected_task = st.selectbox(
                "Select a task to see its metrics:",
                options=top_tasks.tolist(),
                key="auto_task_metrics_selector"
            )
            
            if selected_task:
                task_metrics = filtered_tm[filtered_tm['Task'] == selected_task]
                metrics_for_task = task_metrics['Metric'].value_counts()
                
                st.write(f"**Metrics for {selected_task}:**")
                for i, (metric, count) in enumerate(metrics_for_task.items(), 1):
                    st.write(f"{i}. **{metric}**: {count} papers")
    
    # Advanced task-metric analysis
    if show_advanced:
        st.subheader("🔬 Advanced Task-Metrics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tasks by Metrics Diversity")
            
            # Calculate metrics diversity per task
            task_diversity = tm_df.groupby('Task')['Metric'].nunique().reset_index()
            task_diversity.columns = ['Task', 'Unique_Metrics']
            task_diversity = task_diversity.sort_values('Unique_Metrics', ascending=False).head(10)
            
            if not task_diversity.empty:
                fig_div = px.bar(task_diversity, x='Task', y='Unique_Metrics',
                               title="Tasks with Most Diverse Metrics",
                               text='Unique_Metrics')
                fig_div.update_layout(xaxis_tickangle=-45, height=400)
                fig_div.update_traces(textposition='outside')
                st.plotly_chart(fig_div, use_container_width=True)
        
        with col2:
            st.subheader("Metrics by Task Coverage")
            
            # Calculate task coverage per metric
            metric_coverage = tm_df.groupby('Metric')['Task'].nunique().reset_index()
            metric_coverage.columns = ['Metric', 'Unique_Tasks']
            metric_coverage = metric_coverage.sort_values('Unique_Tasks', ascending=False).head(10)
            
            if not metric_coverage.empty:
                fig_cov = px.bar(metric_coverage, x='Metric', y='Unique_Tasks',
                               title="Metrics Used Across Most Tasks",
                               text='Unique_Tasks')
                fig_cov.update_layout(xaxis_tickangle=-45, height=400)
                fig_cov.update_traces(textposition='outside')
                st.plotly_chart(fig_cov, use_container_width=True)
        
        # Task-Metrics Heatmap
        st.subheader("🗺️ Task-Metrics Heatmap")
        
        # Get top tasks and metrics for heatmap
        top_tasks = tm_df['Task'].value_counts().head(8).index
        top_metrics_for_heatmap = tm_df['Metric'].value_counts().head(12).index
        
        heatmap_data = tm_df[
            (tm_df['Task'].isin(top_tasks)) & 
            (tm_df['Metric'].isin(top_metrics_for_heatmap))
        ]
        
        if not heatmap_data.empty:
            heatmap_counts = heatmap_data.groupby(['Task', 'Metric']).size().reset_index(name='Count')
            heatmap_pivot = heatmap_counts.pivot(index='Task', columns='Metric', values='Count').fillna(0)
            
            fig_heatmap = px.imshow(heatmap_pivot, 
                                  labels=dict(x="Metrics", y="Tasks", color="Usage Count"),
                                  title="Task-Metric Usage Heatmap",
                                  color_continuous_scale="Blues")
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Temporal trends in task-metric combinations
        st.subheader("📈 Task-Metric Evolution Over Time")
        
        # Analyze trends for top 5 task-metric combinations
        # Get the top combinations from the earlier analysis
        tm_combinations = filtered_tm.groupby(['Task', 'Metric']).size().reset_index(name='Count')
        top_5_tm = tm_combinations.nlargest(5, 'Count')
        temporal_data = []
        
        for _, row in top_5_tm.iterrows():
            task, metric = row['Task'], row['Metric']
            task_metric_data = tm_df[
                (tm_df['Task'] == task) & (tm_df['Metric'] == metric)
            ]
            
            for _, tm_row in task_metric_data.iterrows():
                temporal_data.append({
                    'Year': tm_row['Year'],
                    'Task_Metric': f"{task} → {metric}",
                    'Conference': tm_row['Conference']
                })
        
        if temporal_data:
            temp_df = pd.DataFrame(temporal_data)
            yearly_tm = temp_df.groupby(['Year', 'Task_Metric']).size().reset_index(name='Count')
            
            fig_temporal = create_plot(yearly_tm, "Line Chart", 'Year', 'Count', 'Task_Metric',
                                     "Task-Metric Combinations Over Time", False, 400)
            st.plotly_chart(fig_temporal, use_container_width=True)

def show_text_only_analysis(data, paper_subset):
    st.header("📝 Text-Only NLG Tasks Analysis")
    st.markdown(f"Analysis of text-only tasks in **{paper_subset}** (excluding multimodal tasks)")
    
    if paper_subset in ["NLG Text-Only Papers", "NLG Text-Only with All Metrics"]:
        text_only_papers = data  # Already filtered
    else:
        # Filter NLG papers with text-only tasks
        nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes']
        text_only_papers = []
        
        for paper in nlg_papers:
            tasks = paper['answer_1'].get('tasks', [])
            if tasks and is_text_only_task(tasks):
                text_only_papers.append(paper)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_papers = len([p for p in data if p['answer_1']['answer'] == 'Yes']) if paper_subset == "All Papers" else len(data)
        st.metric("Total Papers", f"{total_papers:,}")
    with col2:
        st.metric("Text-Only Papers", f"{len(text_only_papers):,}")
    with col3:
        percentage = (len(text_only_papers) / total_papers * 100) if total_papers > 0 else 0
        st.metric("Text-Only Percentage", f"{percentage:.1f}%")
    
    if not text_only_papers:
        st.warning("No text-only NLG papers in current selection.")
        return
    
    # Extract and clean tasks
    task_counter = Counter()
    for paper in text_only_papers:
        tasks = paper['answer_1'].get('tasks', [])
        for task in tasks:
            cleaned_task = clean_task_name(task)
            task_counter[cleaned_task] += 1
    
    # Get top 10 tasks
    top_tasks = task_counter.most_common(10)
    st.subheader("Top 10 Text-Only NLG Tasks")
    
    if top_tasks:
        task_names = [task for task, count in top_tasks]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tasks_df = pd.DataFrame([
                {'Task': task, 'Count': count}
                for task, count in top_tasks
            ])
            
            plot_type = st.selectbox("Plot Type:", 
                                     ["Horizontal Bar", "Bar Chart", "Treemap", "Sunburst"], 
                                     key="tasks_plot")
            
            fig = create_plot(tasks_df, plot_type, 'Task', 'Count', None,
                            "Most Common Text-Only NLG Tasks", False, 500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Task Details")
            for i, (task, count) in enumerate(top_tasks, 1):
                st.write(f"**{i}. {task}**")
                st.write(f"   Papers: {count}")
        
        # Filter papers with top-10 tasks for evaluation analysis
        top_task_papers = []
        for paper in text_only_papers:
            paper_tasks = [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
            if any(task in task_names for task in paper_tasks):
                top_task_papers.append(paper)
        
        st.subheader(f"Evaluation Approaches in Top-10 Text Tasks ({len(top_task_papers)} papers)")
        
        # Evaluation distribution for top-10 tasks
        eval_counts = Counter()
        for paper in top_task_papers:
            eval_type = get_evaluation_type(paper)
            eval_counts[eval_type] += 1
        
        if eval_counts:
            eval_df = pd.DataFrame([
                {'Evaluation Type': k, 'Count': v} 
                for k, v in eval_counts.items()
            ])
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(eval_df, values='Count', names='Evaluation Type',
                            title="Evaluation Methods for Top-10 Text Tasks")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_plot(eval_df, "Bar Chart", 'Evaluation Type', 'Count', None,
                                "Count by Evaluation Type", False, 400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No tasks found in current selection.")

# Simplified versions of remaining functions for space
def show_tasks_distribution(data, paper_subset):
    st.header("🎯 NLG Tasks Distribution Analysis")
    st.markdown(f"Distribution of NLG tasks in **{paper_subset}**")
    
    # Plot controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", 
                                ["Horizontal Bar", "Bar Chart", "Heatmap", "Treemap", "Line Chart"], 
                                key="tasks_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="tasks_pct")
    with col3:
        analysis_focus = st.selectbox("Analysis Focus:", 
                                    ["Task Overview", "Per-Year Analysis", "Per-Conference Analysis", "Temporal Trends"], 
                                    key="tasks_focus")
    with col4:
        show_advanced = st.checkbox("Advanced Analysis", value=False, key="tasks_advanced",
                                   help="Show detailed task distribution patterns")
    
    nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes'] if paper_subset == "All Papers" else data
    
    if not nlg_papers:
        st.warning("No NLG papers in current selection.")
        return
    
    # Collect all tasks
    all_tasks = []
    for paper in nlg_papers:
        tasks = paper['answer_1'].get('tasks', [])
        for task in tasks:
            cleaned_task = clean_task_name(task)
            all_tasks.append({
                'Task': cleaned_task,
                'Conference': paper['conference'],
                'Year': paper['year'],
                'Paper_ID': paper.get('paper_id', '')
            })
    
    if not all_tasks:
        st.warning("No task data available.")
        return
    
    tasks_df = pd.DataFrame(all_tasks)
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_task_instances = len(all_tasks)
        st.metric("Total Task Instances", f"{total_task_instances:,}")
    with col2:
        unique_tasks = tasks_df['Task'].nunique()
        st.metric("Unique Tasks", f"{unique_tasks:,}")
    with col3:
        avg_tasks = total_task_instances / len(nlg_papers) if nlg_papers else 0
        st.metric("Avg Tasks/Paper", f"{avg_tasks:.1f}")
    
    # Analysis based on focus
    if analysis_focus == "Task Overview":
        show_task_overview_analysis(tasks_df, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Per-Year Analysis":
        show_task_per_year_analysis(tasks_df, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Per-Conference Analysis":
        show_task_per_conference_analysis(tasks_df, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Temporal Trends":
        show_task_temporal_trends_analysis(tasks_df, show_advanced)

def show_task_overview_analysis(tasks_df, plot_type, show_percentage, show_advanced):
    st.subheader("📊 Task Overview Analysis")
    
    # Get top 20 tasks
    task_counts = tasks_df['Task'].value_counts().head(20)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 20 NLG Tasks")
        tasks_overview_df = pd.DataFrame({
            'Task': task_counts.index,
            'Count': task_counts.values
        })
        
        fig = create_plot(tasks_overview_df, plot_type, 'Task', 'Count', None,
                        "Most Popular NLG Tasks", show_percentage, 700)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Tasks")
        for i, (task, count) in enumerate(task_counts.head(10).items(), 1):
            st.write(f"**{i}. {task}**: {count}")
    
    if show_advanced:
        st.subheader("🔬 Advanced Task Analysis")
        
        # Task diversity across conferences
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Task Distribution by Conference")
            
            # Get top 10 tasks for cleaner visualization
            top_tasks = task_counts.head(10).index
            task_conf_data = tasks_df[tasks_df['Task'].isin(top_tasks)]
            
            task_conf_counts = task_conf_data.groupby(['Conference', 'Task']).size().reset_index(name='Count')
            
            if not task_conf_counts.empty:
                fig_conf = create_plot(task_conf_counts, "Bar Chart", 'Conference', 'Count', 'Task',
                                     "Task Distribution Across Conferences", False, 400)
                st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            st.subheader("Conference Task Diversity")
            
            # Calculate unique tasks per conference
            conf_diversity = tasks_df.groupby('Conference')['Task'].nunique().reset_index()
            conf_diversity.columns = ['Conference', 'Unique_Tasks']
            conf_diversity = conf_diversity.sort_values('Unique_Tasks', ascending=False)
            
            if not conf_diversity.empty:
                fig_div = px.bar(conf_diversity, x='Conference', y='Unique_Tasks',
                               title="Unique Tasks per Conference",
                               text='Unique_Tasks')
                fig_div.update_layout(height=400)
                fig_div.update_traces(textposition='outside')
                st.plotly_chart(fig_div, use_container_width=True)

def show_task_per_year_analysis(tasks_df, plot_type, show_percentage, show_advanced):
    st.subheader("📅 Task Distribution Per Year")
    
    # Get top tasks ordered by total count
    task_counts = tasks_df['Task'].value_counts()
    top_tasks = task_counts.head(12).index  # Already ordered by count
    yearly_data = tasks_df[tasks_df['Task'].isin(top_tasks)]
    
    if not yearly_data.empty:
        yearly_counts = yearly_data.groupby(['Year', 'Task']).size().reset_index(name='Count')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if plot_type == "Heatmap":
                # Create year-task heatmap with tasks ordered by total count
                heatmap_pivot = yearly_counts.pivot(index='Task', columns='Year', values='Count').fillna(0)
                # Reorder rows by total count (descending)
                heatmap_pivot = heatmap_pivot.reindex(top_tasks)
                
                fig_heatmap = px.imshow(heatmap_pivot,
                                      labels=dict(x="Year", y="Task", color="Count"),
                                      title="Task Usage by Year (Ordered by Popularity)",
                                      color_continuous_scale="Blues")
                fig_heatmap.update_layout(height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                fig = create_plot(yearly_counts, plot_type, 'Year', 'Count', 'Task',
                                "Task Distribution Over Years", show_percentage, 500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Task Rankings by Year")
            
            # Show ordered task counts
            st.write("**Top Tasks (All Years):**")
            for i, (task, count) in enumerate(task_counts.head(8).items(), 1):
                st.write(f"{i}. **{task}**: {count}")
    
    # Add cross-dimensional analysis
    st.subheader("🔄 Cross-Conference Analysis by Year")
    
    if not yearly_data.empty:
        # Conference-Year-Task analysis
        conf_year_data = yearly_data.groupby(['Conference', 'Year', 'Task']).size().reset_index(name='Count')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Conference Activity Over Years")
            
            # Show how much each conference contributed each year
            conf_year_totals = yearly_data.groupby(['Conference', 'Year']).size().reset_index(name='Total_Tasks')
            conf_year_pivot = conf_year_totals.pivot(index='Conference', columns='Year', values='Total_Tasks').fillna(0)
            
            fig_conf_year = px.imshow(conf_year_pivot,
                                    labels=dict(x="Year", y="Conference", color="Task Instances"),
                                    title="Conference Task Activity by Year",
                                    color_continuous_scale="Oranges")
            fig_conf_year.update_layout(height=400)
            st.plotly_chart(fig_conf_year, use_container_width=True)
        
        with col2:
            st.subheader("Year-Conference Statistics")
            
            # Show yearly breakdown by conference
            for year in sorted(yearly_data['Year'].unique()):
                year_data = yearly_data[yearly_data['Year'] == year]
                conf_counts = year_data.groupby('Conference').size().sort_values(ascending=False)
                
                st.write(f"**{int(year)}:**")
                for conf, count in conf_counts.head(3).items():
                    st.write(f"  • {conf}: {count}")
                st.write("")
    
    if show_advanced:
        st.subheader("🔬 Advanced Yearly Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Task Growth Over Time")
            
            # Calculate cumulative unique tasks
            all_years = sorted(tasks_df['Year'].unique())
            cumulative_tasks = []
            seen_tasks = set()
            
            for year in all_years:
                year_tasks = set(tasks_df[tasks_df['Year'] == year]['Task'].unique())
                seen_tasks.update(year_tasks)
                cumulative_tasks.append({
                    'Year': year,
                    'Cumulative_Unique_Tasks': len(seen_tasks),
                    'New_Tasks_This_Year': len(year_tasks - (seen_tasks - year_tasks))
                })
            
            cumulative_df = pd.DataFrame(cumulative_tasks)
            
            fig_growth = px.line(cumulative_df, x='Year', y='Cumulative_Unique_Tasks',
                               title="Cumulative Unique Tasks Over Time")
            fig_growth.update_layout(height=400)
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            st.subheader("Most Active Years")
            
            year_activity = tasks_df.groupby('Year').size().reset_index(name='Task_Instances')
            year_activity = year_activity.sort_values('Task_Instances', ascending=False)
            
            fig_activity = px.bar(year_activity, x='Year', y='Task_Instances',
                                 title="Task Activity by Year",
                                 text='Task_Instances')
            fig_activity.update_layout(height=400)
            fig_activity.update_traces(textposition='outside')
            st.plotly_chart(fig_activity, use_container_width=True)

def show_task_per_conference_analysis(tasks_df, plot_type, show_percentage, show_advanced):
    st.subheader("🏛️ Task Distribution Per Conference")
    
    # Get top tasks ordered by total count
    task_counts = tasks_df['Task'].value_counts()
    top_tasks = task_counts.head(12).index  # Already ordered by count
    conf_data = tasks_df[tasks_df['Task'].isin(top_tasks)]
    
    if not conf_data.empty:
        conf_counts = conf_data.groupby(['Conference', 'Task']).size().reset_index(name='Count')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if plot_type == "Heatmap":
                # Create conference-task heatmap with tasks ordered by total count
                heatmap_pivot = conf_counts.pivot(index='Task', columns='Conference', values='Count').fillna(0)
                # Reorder rows by total count (descending)
                heatmap_pivot = heatmap_pivot.reindex(top_tasks)
                
                fig_heatmap = px.imshow(heatmap_pivot,
                                      labels=dict(x="Conference", y="Task", color="Count"),
                                      title="Task Usage by Conference (Ordered by Popularity)",
                                      color_continuous_scale="Blues")
                fig_heatmap.update_layout(height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                fig = create_plot(conf_counts, plot_type, 'Conference', 'Count', 'Task',
                                "Task Distribution Across Conferences", show_percentage, 500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Task Rankings by Conference")
            
            # Show ordered task counts
            st.write("**Top Tasks (All Conferences):**")
            for i, (task, count) in enumerate(task_counts.head(8).items(), 1):
                st.write(f"{i}. **{task}**: {count}")
    
    # Add cross-dimensional analysis
    st.subheader("🔄 Cross-Year Analysis by Conference")
    
    if not conf_data.empty:
        # Conference-Year-Task analysis
        conf_year_data = conf_data.groupby(['Conference', 'Year', 'Task']).size().reset_index(name='Count')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Year Activity Across Conferences")
            
            # Show how much each year contributed to each conference
            year_conf_totals = conf_data.groupby(['Year', 'Conference']).size().reset_index(name='Total_Tasks')
            year_conf_pivot = year_conf_totals.pivot(index='Year', columns='Conference', values='Total_Tasks').fillna(0)
            
            fig_year_conf = px.imshow(year_conf_pivot,
                                    labels=dict(x="Conference", y="Year", color="Task Instances"),
                                    title="Year Task Activity by Conference",
                                    color_continuous_scale="Greens")
            fig_year_conf.update_layout(height=400)
            st.plotly_chart(fig_year_conf, use_container_width=True)
        
        with col2:
            st.subheader("Conference-Year Statistics")
            
            # Show conference breakdown by year
            for conf in sorted(conf_data['Conference'].unique()):
                conf_data_subset = conf_data[conf_data['Conference'] == conf]
                year_counts = conf_data_subset.groupby('Year').size().sort_values(ascending=False)
                
                st.write(f"**{conf}:**")
                for year, count in year_counts.head(3).items():
                    st.write(f"  • {int(year)}: {count}")
                st.write("")
    
    if show_advanced:
        st.subheader("🔬 Advanced Conference Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Conference Specialization")
            
            # Find tasks that are predominant in specific conferences
            task_conf_dominance = []
            
            for task in top_tasks[:8]:  # Limit to top 8 for readability
                task_data = tasks_df[tasks_df['Task'] == task]
                conf_counts_for_task = task_data['Conference'].value_counts()
                
                if len(conf_counts_for_task) > 0:
                    dominant_conf = conf_counts_for_task.index[0]
                    dominance_ratio = conf_counts_for_task.iloc[0] / conf_counts_for_task.sum()
                    
                    task_conf_dominance.append({
                        'Task': task,
                        'Dominant_Conference': dominant_conf,
                        'Dominance_Ratio': dominance_ratio
                    })
            
            if task_conf_dominance:
                dominance_df = pd.DataFrame(task_conf_dominance)
                dominance_df = dominance_df.sort_values('Dominance_Ratio', ascending=False)
                
                fig_dom = px.bar(dominance_df, x='Task', y='Dominance_Ratio', 
                               color='Dominant_Conference',
                               title="Task-Conference Dominance",
                               text='Dominant_Conference')
                fig_dom.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_dom, use_container_width=True)
        
        with col2:
            st.subheader("Conference Task Overlap")
            
            # Calculate task overlap between conferences
            conferences = tasks_df['Conference'].unique()
            overlap_data = []
            
            for i, conf1 in enumerate(conferences):
                for conf2 in conferences[i+1:]:
                    conf1_tasks = set(tasks_df[tasks_df['Conference'] == conf1]['Task'])
                    conf2_tasks = set(tasks_df[tasks_df['Conference'] == conf2]['Task'])
                    
                    overlap = len(conf1_tasks.intersection(conf2_tasks))
                    union = len(conf1_tasks.union(conf2_tasks))
                    
                    if union > 0:
                        overlap_data.append({
                            'Conference_Pair': f"{conf1}-{conf2}",
                            'Overlap_Count': overlap,
                            'Overlap_Ratio': overlap / union
                        })
            
            if overlap_data:
                overlap_df = pd.DataFrame(overlap_data)
                overlap_df = overlap_df.sort_values('Overlap_Count', ascending=False)
                
                fig_overlap = px.bar(overlap_df, x='Conference_Pair', y='Overlap_Count',
                                   title="Task Overlap Between Conferences",
                                   text='Overlap_Count')
                fig_overlap.update_layout(xaxis_tickangle=-45, height=400)
                fig_overlap.update_traces(textposition='outside')
                st.plotly_chart(fig_overlap, use_container_width=True)

def show_task_temporal_trends_analysis(tasks_df, show_advanced):
    st.subheader("📈 Temporal Trends Analysis")
    
    # Get top tasks for trend analysis
    top_tasks = tasks_df['Task'].value_counts().head(8).index
    trend_data = tasks_df[tasks_df['Task'].isin(top_tasks)]
    
    if not trend_data.empty:
        # Create temporal trends
        yearly_trends = trend_data.groupby(['Year', 'Task']).size().reset_index(name='Count')
        
        fig_trends = create_plot(yearly_trends, "Line Chart", 'Year', 'Count', 'Task',
                               "Task Popularity Trends Over Time", False, 500)
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Show trending tasks
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Trending Up")
            
            # Find tasks with increasing trends
            trending_up = []
            for task in top_tasks:
                task_yearly = trend_data[trend_data['Task'] == task].groupby('Year').size()
                if len(task_yearly) >= 3:  # Need at least 3 years of data
                    years = task_yearly.index
                    counts = task_yearly.values
                    
                    # Simple trend calculation
                    recent_avg = counts[-2:].mean() if len(counts) >= 2 else counts[-1]
                    early_avg = counts[:2].mean() if len(counts) >= 2 else counts[0]
                    
                    if recent_avg > early_avg:
                        trend_score = (recent_avg - early_avg) / early_avg
                        trending_up.append((task, trend_score))
            
            trending_up.sort(key=lambda x: x[1], reverse=True)
            for task, score in trending_up[:5]:
                st.write(f"• **{task}**: +{score:.1%}")
        
        with col2:
            st.subheader("📉 Trending Down")
            
            # Find tasks with decreasing trends
            trending_down = []
            for task in top_tasks:
                task_yearly = trend_data[trend_data['Task'] == task].groupby('Year').size()
                if len(task_yearly) >= 3:
                    years = task_yearly.index
                    counts = task_yearly.values
                    
                    recent_avg = counts[-2:].mean() if len(counts) >= 2 else counts[-1]
                    early_avg = counts[:2].mean() if len(counts) >= 2 else counts[0]
                    
                    if recent_avg < early_avg:
                        trend_score = (early_avg - recent_avg) / early_avg
                        trending_down.append((task, trend_score))
            
            trending_down.sort(key=lambda x: x[1], reverse=True)
            for task, score in trending_down[:5]:
                st.write(f"• **{task}**: -{score:.1%}")
    
    if show_advanced:
        st.subheader("🔬 Advanced Temporal Analysis")
        
        # Conference-Year-Task 3D analysis
        if not trend_data.empty:
            conf_year_task = trend_data.groupby(['Conference', 'Year', 'Task']).size().reset_index(name='Count')
            
            # Create bubble chart
            fig_3d = px.scatter(
                conf_year_task,
                x='Year',
                y='Conference',
                size='Count',
                color='Task',
                title="Task Distribution: Conference × Year",
                hover_data=['Count'],
                height=500
            )
            st.plotly_chart(fig_3d, use_container_width=True)

def show_llm_evaluation_analysis(data, paper_subset):
    st.header("🧠 LLM Evaluation Analysis")
    st.markdown(f"Comprehensive analysis of LLM evaluation methods, models, and criteria in **{paper_subset}**")
    
    # Plot controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", 
                                ["Horizontal Bar", "Heatmap", "Treemap", "Sunburst", "Scatter Plot"], 
                                key="llm_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="llm_pct")
    with col3:
        analysis_focus = st.selectbox("Analysis Focus:", 
                                    ["Models", "Methods", "Criteria", "Task-Criteria", "Multi-dimensional"], 
                                    key="llm_focus")
    with col4:
        show_advanced = st.checkbox("Advanced Analysis", value=False, key="llm_advanced",
                                   help="Show model-criteria-task relationships and trends")
    
    llm_papers = [p for p in data if p['answer_3']['answer'] == 'Yes']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers in Subset", f"{len(data):,}")
    with col2:
        st.metric("Papers with LLM Eval", f"{len(llm_papers):,}")
    with col3:
        percentage = (len(llm_papers) / len(data) * 100) if data else 0
        st.metric("LLM Evaluation Rate", f"{percentage:.1f}%")
    
    if not llm_papers:
        st.warning("No papers with LLM evaluation in current selection.")
        return
    
    # Collect comprehensive LLM data
    all_models = []
    all_methods = []
    all_criteria = []
    all_combinations = []
    
    # Helper function to filter valid methods (exclude single characters and very short strings)
    def is_valid_method(method):
        if not isinstance(method, str):
            return False
        method = method.strip()
        # Exclude single characters, empty strings, and very short strings
        if len(method) <= 2:
            return False
        # Exclude strings that are just spaces or single characters
        if method in ['i', 'a', 'c', 'n', 's', 'f', 't', 'o', 'r', 'y', 'l', ' ', '']:
            return False
        return True
    
    for paper in llm_papers:
        answer3 = paper['answer_3']
        models = answer3.get('models', [])
        raw_methods = answer3.get('methods', [])
        # Filter out invalid methods
        methods = [m for m in raw_methods if is_valid_method(m)]
        criteria = answer3.get('criteria', [])
        tasks = [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
        
        for model in models:
            all_models.append({
                'Model': model,
                'Conference': paper['conference'],
                'Year': paper['year'],
                'Tasks': tasks,
                'Methods': methods,
                'Criteria': criteria
            })
        
        for method in methods:
            all_methods.append({
                'Method': method,
                'Conference': paper['conference'],
                'Year': paper['year'],
                'Tasks': tasks,
                'Models': models,
                'Criteria': criteria
            })
        
        for criterion in criteria:
            all_criteria.append({
                'Criterion': criterion,
                'Conference': paper['conference'],
                'Year': paper['year'],
                'Tasks': tasks,
                'Models': models,
                'Methods': methods
            })
        
        # Collect combinations for multi-dimensional analysis
        if models and criteria:
            for model in models[:2]:  # Limit to avoid explosion
                for criterion in criteria[:3]:
                    all_combinations.append({
                        'Model': model,
                        'Criterion': criterion,
                        'Conference': paper['conference'],
                        'Year': paper['year'],
                        'Task': tasks[0] if tasks else 'Unknown',
                        'Tasks': tasks,  # Add plural for task-criteria analysis
                        'Criteria': criteria  # Add plural for task-criteria analysis
                    })
    
    # Analysis based on focus
    if analysis_focus == "Models":
        show_llm_models_analysis(all_models, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Methods":
        show_llm_methods_analysis(all_methods, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Criteria":
        show_llm_criteria_analysis(all_criteria, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Task-Criteria":
        # Create task-criteria specific combinations (one per paper)
        task_criteria_combinations = []
        for paper in llm_papers:
            answer3 = paper['answer_3']
            criteria = answer3.get('criteria', [])
            tasks = [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
            
            # Only include papers that have both tasks and criteria
            if tasks and criteria:
                task_criteria_combinations.append({
                    'Tasks': tasks,
                    'Criteria': criteria,
                    'Conference': paper['conference'],
                    'Year': paper['year']
                })
        
        show_llm_task_criteria_analysis(task_criteria_combinations, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Multi-dimensional":
        show_llm_multidimensional_analysis(all_combinations, all_models, all_criteria, show_advanced)

def show_llm_models_analysis(all_models, plot_type, show_percentage, show_advanced):
    st.subheader("🤖 LLM Models Analysis")
    
    if not all_models:
        st.warning("No LLM model data available.")
        return
    
    model_counts = Counter([item['Model'] for item in all_models])
    top_models = model_counts.most_common(20)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Most Used LLM Models")
        
        if top_models:
            models_df = pd.DataFrame([
                {'Model': model, 'Count': count}
                for model, count in top_models
            ])
            
            fig = create_plot(models_df, plot_type, 'Model', 'Count', None,
                            "LLM Models Usage Distribution", show_percentage, 600)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Statistics")
        st.metric("Unique Models", f"{len(model_counts):,}")
        st.metric("Total Model Uses", f"{len(all_models):,}")
        
        st.subheader("Top Models")
        for i, (model, count) in enumerate(top_models[:10], 1):
            st.write(f"**{i}. {model}**: {count}")
    
    # Advanced analysis
    if show_advanced:
        st.subheader("🔬 Advanced Model Analysis")
        
        # Models over time
        models_df = pd.DataFrame(all_models)
        if not models_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Usage Trends")
                
                # Get top 8 models for trend analysis
                top_8_models = [model for model, _ in top_models[:8]]
                trend_data = []
                
                for item in all_models:
                    if item['Model'] in top_8_models:
                        trend_data.append({
                            'Year': item['Year'],
                            'Model': item['Model'],
                            'Conference': item['Conference']
                        })
                
                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    yearly_counts = trend_df.groupby(['Year', 'Model']).size().reset_index(name='Count')
                    
                    fig_trend = create_plot(yearly_counts, "Line Chart", 'Year', 'Count', 'Model',
                                          "LLM Model Usage Over Time", False, 400)
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                st.subheader("Models by Conference")
                
                conf_model_data = []
                for item in all_models:
                    if item['Model'] in top_8_models:
                        conf_model_data.append({
                            'Conference': item['Conference'],
                            'Model': item['Model']
                        })
                
                if conf_model_data:
                    conf_df = pd.DataFrame(conf_model_data)
                    conf_counts = conf_df.groupby(['Conference', 'Model']).size().reset_index(name='Count')
                    
                    fig_conf_heatmap = create_advanced_heatmap(
                        conf_counts, 'Model', 'Conference', 'Count',
                        "Model Usage by Conference", 400
                    )
                    st.plotly_chart(fig_conf_heatmap, use_container_width=True)
            
            # Model-Task relationships
            st.subheader("🎯 Model-Task Relationships")
            
            model_task_data = []
            for item in all_models:
                if item['Model'] in top_8_models:
                    for task in item['Tasks'][:2]:  # Limit tasks
                        if task:
                            model_task_data.append({
                                'Model': item['Model'],
                                'Task': task
                            })
            
            if model_task_data:
                mt_df = pd.DataFrame(model_task_data)
                task_counts = mt_df['Task'].value_counts()
                top_tasks_for_models = task_counts.head(8).index.tolist()
                
                filtered_mt = mt_df[mt_df['Task'].isin(top_tasks_for_models)]
                mt_counts = filtered_mt.groupby(['Task', 'Model']).size().reset_index(name='Count')
                
                if not mt_counts.empty:
                    # Create treemap for model-task hierarchy
                    fig_tree = px.treemap(
                        mt_counts.head(40),
                        path=['Model', 'Task'],
                        values='Count',
                        title="Model → Task Usage Hierarchy"
                    )
                    fig_tree.update_layout(height=500)
                    st.plotly_chart(fig_tree, use_container_width=True)

def show_llm_methods_analysis(all_methods, plot_type, show_percentage, show_advanced):
    st.subheader("⚙️ LLM Methods Analysis")
    
    if not all_methods:
        st.warning("No LLM method data available.")
        return
    
    method_counts = Counter([item['Method'] for item in all_methods])
    top_methods = method_counts.most_common(15)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if top_methods:
            methods_df = pd.DataFrame([
                {'Method': method, 'Count': count}
                for method, count in top_methods
            ])
            
            fig = create_plot(methods_df, plot_type, 'Method', 'Count', None,
                            "LLM Evaluation Methods", show_percentage, 500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Unique Methods", f"{len(method_counts):,}")
        st.metric("Total Method Uses", f"{len(all_methods):,}")
        
        st.subheader("Top Methods")
        for i, (method, count) in enumerate(top_methods[:8], 1):
            st.write(f"**{i}. {method}**: {count}")
    
    if show_advanced:
        st.subheader("🔬 Advanced Methods Analysis")
        
        methods_df = pd.DataFrame(all_methods)
        if not methods_df.empty:
            # Methods evolution
            method_year_data = []
            top_5_methods = [method for method, _ in top_methods[:5]]
            
            for item in all_methods:
                if item['Method'] in top_5_methods:
                    method_year_data.append({
                        'Year': item['Year'],
                        'Method': item['Method'],
                        'Conference': item['Conference']
                    })
            
            if method_year_data:
                my_df = pd.DataFrame(method_year_data)
                yearly_methods = my_df.groupby(['Year', 'Method']).size().reset_index(name='Count')
                
                fig_method_trend = create_plot(yearly_methods, "Line Chart", 'Year', 'Count', 'Method',
                                             "LLM Method Usage Trends", False, 400)
                st.plotly_chart(fig_method_trend, use_container_width=True)

def show_llm_criteria_analysis(all_criteria, plot_type, show_percentage, show_advanced):
    st.subheader("📋 LLM Criteria Analysis")
    
    if not all_criteria:
        st.warning("No LLM criteria data available.")
        return
    
    criteria_counts = Counter([item['Criterion'] for item in all_criteria])
    top_criteria = criteria_counts.most_common(20)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if top_criteria:
            criteria_df = pd.DataFrame([
                {'Criterion': criterion, 'Count': count}
                for criterion, count in top_criteria
            ])
            
            fig = create_plot(criteria_df, plot_type, 'Criterion', 'Count', None,
                            "LLM Evaluation Criteria", show_percentage, 600)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Unique Criteria", f"{len(criteria_counts):,}")
        st.metric("Total Criteria Uses", f"{len(all_criteria):,}")
        
        st.subheader("Top Criteria")
        for i, (criterion, count) in enumerate(top_criteria[:10], 1):
            st.write(f"**{i}. {criterion}**: {count}")
    
    if show_advanced:
        st.subheader("🔬 Advanced Criteria Analysis")
        
        criteria_df = pd.DataFrame(all_criteria)
        if not criteria_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Criteria by conference
                conf_criteria_data = []
                top_8_criteria = [criterion for criterion, _ in top_criteria[:8]]
                
                for item in all_criteria:
                    if item['Criterion'] in top_8_criteria:
                        conf_criteria_data.append({
                            'Conference': item['Conference'],
                            'Criterion': item['Criterion']
                        })
                
                if conf_criteria_data:
                    cc_df = pd.DataFrame(conf_criteria_data)
                    cc_counts = cc_df.groupby(['Conference', 'Criterion']).size().reset_index(name='Count')
                    
                    fig_cc_heatmap = create_advanced_heatmap(
                        cc_counts, 'Criterion', 'Conference', 'Count',
                        "Criteria Usage by Conference", 400
                    )
                    st.plotly_chart(fig_cc_heatmap, use_container_width=True)
            
            with col2:
                # Criteria evolution
                criteria_year_data = []
                for item in all_criteria:
                    if item['Criterion'] in top_8_criteria:
                        criteria_year_data.append({
                            'Year': item['Year'],
                            'Criterion': item['Criterion']
                        })
                
                if criteria_year_data:
                    cy_df = pd.DataFrame(criteria_year_data)
                    yearly_criteria = cy_df.groupby(['Year', 'Criterion']).size().reset_index(name='Count')
                    
                    fig_criteria_trend = create_plot(yearly_criteria, "Line Chart", 'Year', 'Count', 'Criterion',
                                                   "Criteria Usage Trends", False, 400)
                    st.plotly_chart(fig_criteria_trend, use_container_width=True)

def show_llm_task_criteria_analysis(all_combinations, plot_type, show_percentage, show_advanced):
    st.subheader("🎯📋 Task-Criteria Combined Analysis")
    st.markdown("Analyzing the relationship between NLG tasks and LLM evaluation criteria")
    
    if not all_combinations:
        st.warning("No task-criteria data available.")
        return
    
    # Extract task-criteria pairs
    task_criteria_pairs = []
    for combo in all_combinations:
        tasks = combo.get('Tasks', [])
        criteria = combo.get('Criteria', [])
        
        for task in tasks[:3]:  # Limit tasks per paper
            for criterion in criteria[:3]:  # Limit criteria per paper
                if task and criterion:
                    task_criteria_pairs.append({
                        'Task': task,
                        'Criterion': criterion,
                        'Conference': combo['Conference'],
                        'Year': combo['Year']
                    })
    
    if not task_criteria_pairs:
        st.warning("No task-criteria pairs found.")
        return
    
    tc_df = pd.DataFrame(task_criteria_pairs)
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Tasks", tc_df['Task'].nunique())
    with col2:
        st.metric("Unique Criteria", tc_df['Criterion'].nunique())
    with col3:
        st.metric("Task-Criteria Pairs", len(task_criteria_pairs))
    
    # Task-Criteria Relationship Analysis
    st.subheader("📊 LLM Criteria Usage by Task")
    
    # Get top tasks and criteria for cleaner visualization
    top_tasks = tc_df['Task'].value_counts().head(10).index
    top_criteria = tc_df['Criterion'].value_counts().head(12).index
    
    # Filter data for better visualization
    filtered_tc = tc_df[
        (tc_df['Task'].isin(top_tasks)) & 
        (tc_df['Criterion'].isin(top_criteria))
    ]
    
    if not filtered_tc.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if plot_type == "Heatmap":
                # Create criteria-task heatmap (like the human evaluation example)
                tc_counts = filtered_tc.groupby(['Criterion', 'Task']).size().reset_index(name='Count')
                heatmap_pivot = tc_counts.pivot(index='Criterion', columns='Task', values='Count').fillna(0)
                
                fig_heatmap = px.imshow(heatmap_pivot,
                                      labels=dict(x="Tasks", y="LLM Evaluation Criteria", color="Usage Count"),
                                      title="LLM Evaluation Criteria by NLG Task",
                                      color_continuous_scale="Blues")
                fig_heatmap.update_layout(height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                # Use tasks as color indicators for other plot types
                criteria_counts = filtered_tc.groupby(['Criterion', 'Task']).size().reset_index(name='Count')
                
                fig = create_plot(criteria_counts, plot_type, 'Criterion', 'Count', 'Task',
                                "LLM Criteria Usage with Task Indicators", show_percentage, 500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Criteria-Task Summary")
            
            # Show most used criteria
            criteria_totals = filtered_tc.groupby('Criterion').size().sort_values(ascending=False)
            st.write("**Most Used Criteria:**")
            for i, (criterion, count) in enumerate(criteria_totals.head(8).items(), 1):
                st.write(f"{i}. **{criterion}**: {count}")
    
    # Additional Analysis: Criteria Distribution Across Tasks
    st.subheader("🎯 Criteria Distribution Analysis")
    
    if not filtered_tc.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Evaluated Tasks")
            
            # Show tasks with criteria usage, using tasks as the main dimension
            task_criteria_summary = filtered_tc.groupby(['Task', 'Criterion']).size().reset_index(name='Count')
            task_totals = task_criteria_summary.groupby('Task')['Count'].sum().sort_values(ascending=False)
            
            # Create bar chart with task totals
            task_totals_df = pd.DataFrame({
                'Task': task_totals.index,
                'Total_Criteria_Usage': task_totals.values
            })
            
            fig_tasks = px.bar(task_totals_df, x='Task', y='Total_Criteria_Usage',
                             title="Tasks by LLM Criteria Usage",
                             text='Total_Criteria_Usage')
            fig_tasks.update_layout(xaxis_tickangle=-45, height=400)
            fig_tasks.update_traces(textposition='outside')
            st.plotly_chart(fig_tasks, use_container_width=True)
        
        with col2:
            st.subheader("Task-Specific Criteria")
            
            # Interactive task selector
            selected_task = st.selectbox(
                "Select a task to see its criteria:",
                options=top_tasks.tolist(),
                key="llm_task_criteria_selector"
            )
            
            if selected_task:
                task_criteria = filtered_tc[filtered_tc['Task'] == selected_task]
                criteria_for_task = task_criteria['Criterion'].value_counts()
                
                st.write(f"**Criteria for {selected_task}:**")
                for i, (criterion, count) in enumerate(criteria_for_task.items(), 1):
                    st.write(f"{i}. **{criterion}**: {count} papers")
    
    # Task-based criteria analysis
    if show_advanced:
        st.subheader("🔬 Advanced Task-Criteria Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tasks by Criteria Diversity")
            
            # Calculate criteria diversity per task
            task_diversity = tc_df.groupby('Task')['Criterion'].nunique().reset_index()
            task_diversity.columns = ['Task', 'Unique_Criteria']
            task_diversity = task_diversity.sort_values('Unique_Criteria', ascending=False).head(10)
            
            if not task_diversity.empty:
                fig_div = px.bar(task_diversity, x='Task', y='Unique_Criteria',
                               title="Tasks with Most Diverse Criteria",
                               text='Unique_Criteria')
                fig_div.update_layout(xaxis_tickangle=-45, height=400)
                fig_div.update_traces(textposition='outside')
                st.plotly_chart(fig_div, use_container_width=True)
        
        with col2:
            st.subheader("Criteria by Task Coverage")
            
            # Calculate task coverage per criterion
            criteria_coverage = tc_df.groupby('Criterion')['Task'].nunique().reset_index()
            criteria_coverage.columns = ['Criterion', 'Unique_Tasks']
            criteria_coverage = criteria_coverage.sort_values('Unique_Tasks', ascending=False).head(10)
            
            if not criteria_coverage.empty:
                fig_cov = px.bar(criteria_coverage, x='Criterion', y='Unique_Tasks',
                               title="Criteria Used Across Most Tasks",
                               text='Unique_Tasks')
                fig_cov.update_layout(xaxis_tickangle=-45, height=400)
                fig_cov.update_traces(textposition='outside')
                st.plotly_chart(fig_cov, use_container_width=True)
        
        # Task-Criteria Heatmap
        st.subheader("🗺️ Task-Criteria Heatmap")
        
        # Get top tasks and criteria for heatmap
        top_tasks = tc_df['Task'].value_counts().head(8).index
        top_criteria = tc_df['Criterion'].value_counts().head(10).index
        
        heatmap_data = tc_df[
            (tc_df['Task'].isin(top_tasks)) & 
            (tc_df['Criterion'].isin(top_criteria))
        ]
        
        if not heatmap_data.empty:
            heatmap_counts = heatmap_data.groupby(['Task', 'Criterion']).size().reset_index(name='Count')
            heatmap_pivot = heatmap_counts.pivot(index='Task', columns='Criterion', values='Count').fillna(0)
            
            fig_heatmap = px.imshow(heatmap_pivot, 
                                  labels=dict(x="Criteria", y="Tasks", color="Usage Count"),
                                  title="Task-Criteria Usage Heatmap",
                                  color_continuous_scale="Blues")
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Temporal trends in task-criteria combinations
        st.subheader("📈 Task-Criteria Evolution Over Time")
        
        # Analyze trends for top 5 task-criteria combinations
        # Get the top combinations from the earlier analysis
        tc_combinations = filtered_tc.groupby(['Task', 'Criterion']).size().reset_index(name='Count')
        top_5_tc = tc_combinations.nlargest(5, 'Count')
        temporal_data = []
        
        for _, row in top_5_tc.iterrows():
            task, criterion = row['Task'], row['Criterion']
            task_criterion_data = tc_df[
                (tc_df['Task'] == task) & (tc_df['Criterion'] == criterion)
            ]
            
            for _, tc_row in task_criterion_data.iterrows():
                temporal_data.append({
                    'Year': tc_row['Year'],
                    'Task_Criterion': f"{task} → {criterion}",
                    'Conference': tc_row['Conference']
                })
        
        if temporal_data:
            temp_df = pd.DataFrame(temporal_data)
            yearly_tc = temp_df.groupby(['Year', 'Task_Criterion']).size().reset_index(name='Count')
            
            fig_temporal = create_plot(yearly_tc, "Line Chart", 'Year', 'Count', 'Task_Criterion',
                                     "Task-Criteria Combinations Over Time", False, 400)
            st.plotly_chart(fig_temporal, use_container_width=True)

def show_llm_multidimensional_analysis(all_combinations, all_models, all_criteria, show_advanced):
    st.subheader("🌐 Multi-Dimensional LLM Analysis")
    
    if not all_combinations:
        st.warning("No multi-dimensional data available.")
        return
    
    # Model-Criteria Relationship Analysis
    st.subheader("🔗 LLM Models and Criteria Analysis")
    
    # Create DataFrame for model-criteria relationships
    combo_df = pd.DataFrame(all_combinations)
    
    if not combo_df.empty:
        # Get top models and criteria for cleaner visualization
        top_models = combo_df['Model'].value_counts().head(10).index
        top_criteria = combo_df['Criterion'].value_counts().head(12).index
        
        # Filter for better visualization
        filtered_combo = combo_df[
            (combo_df['Model'].isin(top_models)) & 
            (combo_df['Criterion'].isin(top_criteria))
        ]
        
        if not filtered_combo.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Models and Criteria Heatmap")
                
                # Create model-criteria heatmap
                mc_counts = filtered_combo.groupby(['Model', 'Criterion']).size().reset_index(name='Count')
                heatmap_pivot = mc_counts.pivot(index='Model', columns='Criterion', values='Count').fillna(0)
                
                fig_heatmap = px.imshow(heatmap_pivot,
                                      labels=dict(x="LLM Evaluation Criteria", y="LLM Models", color="Usage Count"),
                                      title="LLM Models × Criteria Usage Heatmap",
                                      color_continuous_scale="Blues")
                fig_heatmap.update_layout(height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                st.subheader("Model-Criteria Statistics")
                
                # Show most active models
                model_totals = filtered_combo.groupby('Model').size().sort_values(ascending=False)
                st.write("**Most Active Models:**")
                for i, (model, count) in enumerate(model_totals.head(6).items(), 1):
                    st.write(f"{i}. **{model}**: {count}")
                
                st.write("**Most Used Criteria:**")
                criteria_totals = filtered_combo.groupby('Criterion').size().sort_values(ascending=False)
                for i, (criterion, count) in enumerate(criteria_totals.head(6).items(), 1):
                    st.write(f"{i}. **{criterion}**: {count}")
        
        # Additional Analysis: Separate Dimensions
        st.subheader("📊 Model and Criteria Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Models by Criteria Diversity")
            
            # Calculate criteria diversity per model
            model_diversity = filtered_combo.groupby('Model')['Criterion'].nunique().reset_index()
            model_diversity.columns = ['Model', 'Unique_Criteria']
            model_diversity = model_diversity.sort_values('Unique_Criteria', ascending=False)
            
            if not model_diversity.empty:
                fig_model_div = px.bar(model_diversity, x='Model', y='Unique_Criteria',
                                     title="Models with Most Diverse Criteria",
                                     text='Unique_Criteria')
                fig_model_div.update_layout(xaxis_tickangle=-45, height=400)
                fig_model_div.update_traces(textposition='outside')
                st.plotly_chart(fig_model_div, use_container_width=True)
        
        with col2:
            st.subheader("Interactive Model Selector")
            
            # Model selector for detailed analysis
            selected_model = st.selectbox(
                "Select a model to see its criteria:",
                options=top_models.tolist(),
                key="llm_model_criteria_selector"
            )
            
            if selected_model:
                model_criteria = filtered_combo[filtered_combo['Model'] == selected_model]
                criteria_for_model = model_criteria['Criterion'].value_counts()
                
                st.write(f"**Criteria for {selected_model}:**")
                for i, (criterion, count) in enumerate(criteria_for_model.items(), 1):
                    st.write(f"{i}. **{criterion}**: {count} papers")
    
    if show_advanced:
        # Complex network analysis
        st.subheader("🕸️ Model-Criteria Network Analysis")
        
        combo_df = pd.DataFrame(all_combinations)
        if not combo_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Model-Criteria scatter plot (network-like)
                mc_summary = combo_df.groupby(['Model', 'Criterion']).size().reset_index(name='Count')
                model_counts = combo_df['Model'].value_counts()
                criteria_counts = combo_df['Criterion'].value_counts()
                
                # Filter to top items for cleaner visualization
                top_models = model_counts.head(8).index
                top_criteria = criteria_counts.head(10).index
                
                filtered_mc = mc_summary[
                    (mc_summary['Model'].isin(top_models)) &
                    (mc_summary['Criterion'].isin(top_criteria))
                ]
                
                if not filtered_mc.empty:
                    fig_network = px.scatter(
                        filtered_mc,
                        x='Model',
                        y='Criterion',
                        size='Count',
                        color='Count',
                        title="Model-Criteria Co-occurrence Network",
                        color_continuous_scale='Blues',
                        height=500
                    )
                    fig_network.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_network, use_container_width=True)
            
            with col2:
                # Conference-Year-Model bubble chart
                conf_year_data = combo_df.groupby(['Conference', 'Year', 'Model']).size().reset_index(name='Count')
                
                if not conf_year_data.empty:
                    fig_bubble = px.scatter(
                        conf_year_data,
                        x='Year',
                        y='Conference',
                        size='Count',
                        color='Model',
                        title="Model Usage: Conference × Year",
                        height=500
                    )
                    st.plotly_chart(fig_bubble, use_container_width=True)
            
            # Task-Model-Criteria sunburst
            st.subheader("☀️ Task-Model-Criteria Hierarchy")
            
            task_model_criteria = []
            for combo in all_combinations[:100]:  # Limit for performance
                if combo['Task'] and combo['Task'] != 'Unknown':
                    task_model_criteria.append({
                        'Task': combo['Task'],
                        'Model': combo['Model'],
                        'Criterion': combo['Criterion']
                    })
            
            if task_model_criteria:
                tmc_df = pd.DataFrame(task_model_criteria)
                tmc_counts = tmc_df.groupby(['Task', 'Model', 'Criterion']).size().reset_index(name='Count')
                
                # Get top combinations
                top_tmc = tmc_counts.nlargest(30, 'Count')
                
                if not top_tmc.empty:
                    fig_sunburst = px.sunburst(
                        top_tmc,
                        path=['Task', 'Model', 'Criterion'],
                        values='Count',
                        title="Task → Model → Criteria Hierarchy"
                    )
                    fig_sunburst.update_layout(height=600)
                    st.plotly_chart(fig_sunburst, use_container_width=True)

def show_human_evaluation_analysis(data, paper_subset):
    st.header("👥 Human Evaluation Analysis")
    st.markdown(f"Comprehensive analysis of human evaluation guidelines, criteria, and practices in **{paper_subset}**")
    
    # Plot controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", 
                                ["Horizontal Bar", "Heatmap", "Treemap", "Sunburst", "Scatter Plot"], 
                                key="human_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="human_pct")
    with col3:
        analysis_focus = st.selectbox("Analysis Focus:", 
                                    ["Criteria", "Guidelines", "Task-Criteria", "Temporal Trends"], 
                                    key="human_focus")
    with col4:
        show_advanced = st.checkbox("Advanced Analysis", value=False, key="human_advanced",
                                   help="Show criteria-task relationships and conference comparisons")
    
    human_papers = [p for p in data if p['answer_4']['answer'] == 'Yes']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers in Subset", f"{len(data):,}")
    with col2:
        st.metric("Papers with Human Eval", f"{len(human_papers):,}")
    with col3:
        percentage = (len(human_papers) / len(data) * 100) if data else 0
        st.metric("Human Evaluation Rate", f"{percentage:.1f}%")
    
    if not human_papers:
        st.warning("No papers with human evaluation in current selection.")
        return
    
    # Collect comprehensive human evaluation data
    all_criteria = []
    all_guidelines = []
    criteria_task_combinations = []
    
    for paper in human_papers:
        answer4 = paper['answer_4']
        criteria = answer4.get('criteria', [])
        guideline = answer4.get('guideline', '').strip()
        tasks = [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
        
        for criterion in criteria:
            all_criteria.append({
                'Criterion': criterion,
                'Conference': paper['conference'],
                'Year': paper['year'],
                'Tasks': tasks,
                'Guideline': guideline
            })
        
        if guideline:
            all_guidelines.append({
                'Guideline': guideline,
                'Conference': paper['conference'],
                'Year': paper['year'],
                'Tasks': tasks,
                'Criteria': criteria,
                'Length': len(guideline)
            })
        
        # Create criteria-task combinations
        if criteria and tasks:
            for criterion in criteria[:3]:  # Limit to avoid explosion
                for task in tasks[:2]:
                    criteria_task_combinations.append({
                        'Criterion': criterion,
                        'Task': task,
                        'Conference': paper['conference'],
                        'Year': paper['year']
                    })
    
    # Analysis based on focus
    if analysis_focus == "Criteria":
        show_human_criteria_analysis(all_criteria, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Guidelines":
        show_human_guidelines_analysis(all_guidelines, show_advanced)
    elif analysis_focus == "Task-Criteria":
        show_human_task_criteria_analysis(criteria_task_combinations, plot_type, show_advanced)
    elif analysis_focus == "Temporal Trends":
        show_human_temporal_analysis(all_criteria, all_guidelines, show_advanced)

def show_human_criteria_analysis(all_criteria, plot_type, show_percentage, show_advanced):
    st.subheader("📋 Human Evaluation Criteria Analysis")
    
    if not all_criteria:
        st.warning("No human criteria data available.")
        return
    
    criteria_counts = Counter([item['Criterion'] for item in all_criteria])
    top_criteria = criteria_counts.most_common(25)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if top_criteria:
            criteria_df = pd.DataFrame([
                {'Criterion': criterion, 'Count': count}
                for criterion, count in top_criteria
            ])
            
            fig = create_plot(criteria_df, plot_type, 'Criterion', 'Count', None,
                            "Most Used Human Evaluation Criteria", show_percentage, 700)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Criteria Statistics")
        st.metric("Unique Criteria", f"{len(criteria_counts):,}")
        st.metric("Total Criteria Uses", f"{len(all_criteria):,}")
        avg_criteria = len(all_criteria) / len(set(item['Conference'] + str(item['Year']) for item in all_criteria))
        st.metric("Avg Criteria/Study", f"{avg_criteria:.1f}")
        
        st.subheader("Top Criteria")
        for i, (criterion, count) in enumerate(top_criteria[:12], 1):
            st.write(f"**{i}. {criterion}**: {count}")
    
    if show_advanced:
        st.subheader("🔬 Advanced Criteria Analysis")
        
        criteria_df = pd.DataFrame(all_criteria)
        if not criteria_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Criteria by Conference")
                
                # Conference-criteria heatmap
                conf_criteria_data = []
                top_12_criteria = [criterion for criterion, _ in top_criteria[:12]]
                
                for item in all_criteria:
                    if item['Criterion'] in top_12_criteria:
                        conf_criteria_data.append({
                            'Conference': item['Conference'],
                            'Criterion': item['Criterion']
                        })
                
                if conf_criteria_data:
                    cc_df = pd.DataFrame(conf_criteria_data)
                    cc_counts = cc_df.groupby(['Conference', 'Criterion']).size().reset_index(name='Count')
                    
                    fig_conf_heatmap = create_advanced_heatmap(
                        cc_counts, 'Criterion', 'Conference', 'Count',
                        "Criteria Usage by Conference", 500
                    )
                    st.plotly_chart(fig_conf_heatmap, use_container_width=True)
            
            with col2:
                st.subheader("Criteria Evolution")
                
                # Yearly trends
                criteria_year_data = []
                top_8_criteria = [criterion for criterion, _ in top_criteria[:8]]
                
                for item in all_criteria:
                    if item['Criterion'] in top_8_criteria:
                        criteria_year_data.append({
                            'Year': item['Year'],
                            'Criterion': item['Criterion']
                        })
                
                if criteria_year_data:
                    cy_df = pd.DataFrame(criteria_year_data)
                    yearly_criteria = cy_df.groupby(['Year', 'Criterion']).size().reset_index(name='Count')
                    
                    fig_criteria_trend = create_plot(yearly_criteria, "Line Chart", 'Year', 'Count', 'Criterion',
                                                   "Human Criteria Trends Over Time", False, 500)
                    st.plotly_chart(fig_criteria_trend, use_container_width=True)
            
            # Criteria co-occurrence analysis
            st.subheader("🔗 Criteria Co-occurrence Analysis")
            
            # Group by paper and find criteria combinations
            paper_criteria = defaultdict(set)
            for item in all_criteria:
                paper_id = f"{item['Conference']}_{item['Year']}"
                paper_criteria[paper_id].add(item['Criterion'])
            
            cooccurrence_data = []
            for paper_crits in paper_criteria.values():
                paper_crits_list = list(paper_crits)
                for i, crit1 in enumerate(paper_crits_list):
                    for crit2 in paper_crits_list[i+1:]:
                        if crit1 in top_12_criteria and crit2 in top_12_criteria:
                            cooccurrence_data.append({
                                'Criterion1': crit1,
                                'Criterion2': crit2
                            })
            
            if cooccurrence_data:
                cooc_df = pd.DataFrame(cooccurrence_data)
                cooc_counts = cooc_df.groupby(['Criterion1', 'Criterion2']).size().reset_index(name='Count')
                
                if not cooc_counts.empty:
                    # Network-style scatter plot
                    fig_network = px.scatter(
                        cooc_counts,
                        x='Criterion1',
                        y='Criterion2',
                        size='Count',
                        color='Count',
                        title="Criteria Co-occurrence Network",
                        color_continuous_scale='Reds',
                        height=500
                    )
                    fig_network.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_network, use_container_width=True)
                    
                    # Show top combinations
                    st.write("**Most Common Criteria Combinations:**")
                    top_pairs = cooc_counts.nlargest(10, 'Count')
                    for _, row in top_pairs.iterrows():
                        st.write(f"• **{row['Criterion1']}** + **{row['Criterion2']}**: {row['Count']} papers")

def show_human_guidelines_analysis(all_guidelines, show_advanced):
    st.subheader("📜 Human Evaluation Guidelines Analysis")
    
    if not all_guidelines:
        st.warning("No human guidelines data available.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers with Guidelines", f"{len(all_guidelines):,}")
    with col2:
        avg_length = sum(item['Length'] for item in all_guidelines) / len(all_guidelines)
        st.metric("Avg Guideline Length", f"{avg_length:.0f} chars")
    with col3:
        guidelines_by_conf = Counter([item['Conference'] for item in all_guidelines])
        st.metric("Most Guidelines", f"{guidelines_by_conf.most_common(1)[0][0]}")
    
    # Guidelines length distribution
    st.subheader("📏 Guidelines Length Analysis")
    
    lengths_df = pd.DataFrame([
        {'Length': item['Length'], 'Conference': item['Conference'], 'Year': item['Year']}
        for item in all_guidelines
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot of lengths by conference
        fig_box = px.box(lengths_df, x='Conference', y='Length',
                        title="Guidelines Length Distribution by Conference")
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Scatter plot of lengths over time
        fig_scatter = px.scatter(lengths_df, x='Year', y='Length', color='Conference',
                               title="Guidelines Length Trends Over Time",
                               size_max=15)
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    if show_advanced:
        # Guidelines by year and conference heatmap
        st.subheader("🔥 Guidelines Heatmap Analysis")
        
        guidelines_summary = lengths_df.groupby(['Conference', 'Year']).agg({
            'Length': ['count', 'mean']
        }).round(0)
        guidelines_summary.columns = ['Count', 'Avg_Length']
        guidelines_summary = guidelines_summary.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_count_heatmap = create_advanced_heatmap(
                guidelines_summary, 'Year', 'Conference', 'Count',
                "Guidelines Count by Conference-Year", 400
            )
            st.plotly_chart(fig_count_heatmap, use_container_width=True)
        
        with col2:
            fig_length_heatmap = create_advanced_heatmap(
                guidelines_summary, 'Year', 'Conference', 'Avg_Length',
                "Average Guidelines Length", 400
            )
            st.plotly_chart(fig_length_heatmap, use_container_width=True)
        
        # Sample guidelines by category
        st.subheader("📖 Sample Guidelines by Category")
        
        # Categorize guidelines by length
        short_guidelines = [g for g in all_guidelines if g['Length'] < 200]
        medium_guidelines = [g for g in all_guidelines if 200 <= g['Length'] < 500]
        long_guidelines = [g for g in all_guidelines if g['Length'] >= 500]
        
        tab1, tab2, tab3 = st.tabs([f"Short (<200 chars) - {len(short_guidelines)}", 
                                   f"Medium (200-500 chars) - {len(medium_guidelines)}", 
                                   f"Long (>500 chars) - {len(long_guidelines)}"])
        
        with tab1:
            if short_guidelines:
                sample = short_guidelines[:3]
                for i, guideline in enumerate(sample, 1):
                    with st.expander(f"Sample {i}: {guideline['Conference']} {guideline['Year']}"):
                        st.write(guideline['Guideline'])
        
        with tab2:
            if medium_guidelines:
                sample = medium_guidelines[:3]
                for i, guideline in enumerate(sample, 1):
                    with st.expander(f"Sample {i}: {guideline['Conference']} {guideline['Year']}"):
                        st.write(guideline['Guideline'])
        
        with tab3:
            if long_guidelines:
                sample = long_guidelines[:3]
                for i, guideline in enumerate(sample, 1):
                    with st.expander(f"Sample {i}: {guideline['Conference']} {guideline['Year']}"):
                        st.write(guideline['Guideline'][:300] + "..." if len(guideline['Guideline']) > 300 else guideline['Guideline'])

def show_human_task_criteria_analysis(criteria_task_combinations, plot_type, show_advanced):
    st.subheader("🎯 Task-Criteria Relationship Analysis")
    
    if not criteria_task_combinations:
        st.warning("No task-criteria data available.")
        return
    
    # Create task-criteria matrix
    tc_df = pd.DataFrame(criteria_task_combinations)
    task_criteria_counts = tc_df.groupby(['Task', 'Criterion']).size().reset_index(name='Count')
    
    # Get top tasks and criteria for cleaner visualization
    task_counts = tc_df['Task'].value_counts()
    criteria_counts = tc_df['Criterion'].value_counts()
    
    top_10_tasks = task_counts.head(10).index
    top_15_criteria = criteria_counts.head(15).index
    
    filtered_tc = task_criteria_counts[
        (task_criteria_counts['Task'].isin(top_10_tasks)) &
        (task_criteria_counts['Criterion'].isin(top_15_criteria))
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not filtered_tc.empty:
            st.subheader("Task × Criteria Heatmap")
            
            fig_heatmap = create_advanced_heatmap(
                filtered_tc, 'Criterion', 'Task', 'Count',
                "Human Evaluation Criteria by NLG Task", 600
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.subheader("Top Combinations")
        top_combinations = task_criteria_counts.nlargest(15, 'Count')
        
        for _, row in top_combinations.iterrows():
            st.write(f"**{row['Task'][:20]}...** × **{row['Criterion']}**: {row['Count']}")
    
    if show_advanced:
        st.subheader("🔍 Advanced Task-Criteria Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Treemap showing task → criteria hierarchy
            if not filtered_tc.empty:
                fig_treemap = px.treemap(
                    filtered_tc.head(40),
                    path=['Task', 'Criterion'],
                    values='Count',
                    title="Task → Criteria Hierarchy"
                )
                fig_treemap.update_layout(height=500)
                st.plotly_chart(fig_treemap, use_container_width=True)
        
        with col2:
            # Sunburst chart
            if not filtered_tc.empty:
                fig_sunburst = px.sunburst(
                    filtered_tc.head(40),
                    path=['Task', 'Criterion'],
                    values='Count',
                    title="Task-Criteria Sunburst"
                )
                fig_sunburst.update_layout(height=500)
                st.plotly_chart(fig_sunburst, use_container_width=True)

def show_human_temporal_analysis(all_criteria, all_guidelines, show_advanced):
    st.subheader("📈 Temporal Trends in Human Evaluation")
    
    if not all_criteria:
        st.warning("No temporal data available.")
        return
    
    # Yearly statistics
    criteria_df = pd.DataFrame(all_criteria)
    yearly_stats = criteria_df.groupby('Year').agg({
        'Criterion': 'count',
        'Conference': lambda x: len(set(x))
    }).reset_index()
    yearly_stats.columns = ['Year', 'Criteria_Count', 'Conference_Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Human Evaluation Activity Over Time")
        
        fig_activity = px.line(yearly_stats, x='Year', y='Criteria_Count',
                             title="Human Evaluation Criteria Usage Over Time",
                             markers=True)
        fig_activity.update_layout(height=400)
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col2:
        # Guidelines trends if available
        if all_guidelines:
            guidelines_df = pd.DataFrame(all_guidelines)
            yearly_guidelines = guidelines_df.groupby('Year').agg({
                'Length': 'mean',
                'Guideline': 'count'
            }).reset_index()
            yearly_guidelines.columns = ['Year', 'Avg_Length', 'Count']
            
            fig_guidelines = px.line(yearly_guidelines, x='Year', y='Count',
                                   title="Guidelines Usage Over Time",
                                   markers=True)
            fig_guidelines.update_layout(height=400)
            st.plotly_chart(fig_guidelines, use_container_width=True)
    
    if show_advanced:
        st.subheader("🔍 Detailed Temporal Analysis")
        
        # Conference evolution
        conf_year_stats = criteria_df.groupby(['Conference', 'Year']).size().reset_index(name='Count')
        
        fig_conf_evolution = px.line(conf_year_stats, x='Year', y='Count', color='Conference',
                                   title="Human Evaluation by Conference Over Time",
                                   markers=True)
        fig_conf_evolution.update_layout(height=400)
        st.plotly_chart(fig_conf_evolution, use_container_width=True)
        
        # Show yearly statistics table
        st.subheader("📊 Yearly Statistics Summary")
        
        summary_stats = criteria_df.groupby('Year').agg({
            'Criterion': ['count', 'nunique'],
            'Conference': 'nunique'
        }).round(2)
        summary_stats.columns = ['Total_Criteria_Uses', 'Unique_Criteria', 'Conferences']
        
        st.dataframe(summary_stats, use_container_width=True)

def show_cross_evaluation_analysis(data, paper_subset):
    st.header("🔀 Cross-Evaluation Comparison")
    st.markdown(f"Comprehensive analysis of papers using multiple evaluation types in **{paper_subset}**")
    
    # Plot controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", 
                                ["Bar Chart", "Heatmap", "Sunburst", "Treemap", "Scatter Plot"], 
                                key="cross_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="cross_pct")
    with col3:
        analysis_focus = st.selectbox("Analysis Focus:", 
                                    ["Evaluation Combinations", "Criteria Comparison", "Task-based Analysis", "Multi-dimensional", "Temporal Analysis"], 
                                    key="cross_focus")
    with col4:
        show_advanced = st.checkbox("Advanced Analysis", value=False, key="cross_advanced",
                                   help="Show detailed cross-evaluation relationships and patterns")
    
    # Create comprehensive evaluation analysis
    eval_data = []
    for paper in data:
        automatic = paper['answer_2']['answer'] == 'Yes'
        llm = paper['answer_3']['answer'] == 'Yes'
        human = paper['answer_4']['answer'] == 'Yes'
        
        eval_type = get_evaluation_type(paper)
        
        eval_data.append({
            'Paper': paper,
            'Automatic': automatic,
            'LLM': llm,
            'Human': human,
            'EvaluationType': eval_type,
            'Conference': paper['conference'],
            'Year': paper['year'],
            'Tasks': [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])],
            'AutoMetrics': paper['answer_2'].get('automatic_metrics', []),
            'LLMModels': paper['answer_3'].get('models', []),
            'LLMCriteria': paper['answer_3'].get('criteria', []),
            'HumanCriteria': paper['answer_4'].get('criteria', [])
        })
    
    # Analysis based on focus
    if analysis_focus == "Evaluation Combinations":
        show_evaluation_combinations_analysis(eval_data, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Criteria Comparison":
        show_criteria_comparison_analysis(eval_data, plot_type, show_advanced)
    elif analysis_focus == "Task-based Analysis":
        show_task_based_cross_analysis(eval_data, plot_type, show_percentage, show_advanced)
    elif analysis_focus == "Multi-dimensional":
        show_multidimensional_cross_analysis(eval_data, show_advanced)
    elif analysis_focus == "Temporal Analysis":
        show_temporal_cross_analysis(eval_data, show_advanced)

def show_evaluation_combinations_analysis(eval_data, plot_type, show_percentage, show_advanced):
    st.subheader("🎯 Evaluation Method Combinations")
    
    # Basic statistics
    eval_type_counts = Counter([item['EvaluationType'] for item in eval_data])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", f"{len(eval_data):,}")
    with col2:
        triple_count = eval_type_counts['All Three']
        st.metric("All Three Methods", f"{triple_count:,}")
    with col3:
        auto_human = eval_type_counts['Automatic + Human']
        st.metric("Auto + Human", f"{auto_human:,}")
    with col4:
        llm_human = eval_type_counts['LLM + Human']
        st.metric("LLM + Human", f"{llm_human:,}")
    
    # Main visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Evaluation Combination Distribution")
        
        combo_df = pd.DataFrame([
            {'Combination': eval_type, 'Count': count}
            for eval_type, count in eval_type_counts.items()
            if eval_type != 'None'  # Exclude papers with no evaluation
        ])
        
        if not combo_df.empty:
            fig = create_plot(combo_df, plot_type, 'Combination', 'Count', None,
                            "Distribution of Evaluation Method Combinations", show_percentage, 500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Combination Statistics")
        
        for eval_type, count in eval_type_counts.most_common():
            if eval_type != 'None':
                percentage = (count / len(eval_data) * 100)
                st.write(f"**{eval_type}**: {count:,} ({percentage:.1f}%)")
    
    if show_advanced:
        st.subheader("🔬 Advanced Combination Analysis")
        
        # Conference-wise combination analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Combinations by Conference")
            
            conf_combo_data = []
            for item in eval_data:
                if item['EvaluationType'] != 'None':
                    conf_combo_data.append({
                        'Conference': item['Conference'],
                        'Combination': item['EvaluationType']
                    })
            
            if conf_combo_data:
                cc_df = pd.DataFrame(conf_combo_data)
                cc_counts = cc_df.groupby(['Conference', 'Combination']).size().reset_index(name='Count')
                
                fig_conf_heatmap = create_advanced_heatmap(
                    cc_counts, 'Combination', 'Conference', 'Count',
                    "Evaluation Combinations by Conference", 400
                )
                st.plotly_chart(fig_conf_heatmap, use_container_width=True)
        
        with col2:
            st.subheader("Temporal Evolution")
            
            year_combo_data = []
            top_5_combos = [combo for combo, _ in eval_type_counts.most_common(5) if combo != 'None']
            
            for item in eval_data:
                if item['EvaluationType'] in top_5_combos:
                    year_combo_data.append({
                        'Year': item['Year'],
                        'Combination': item['EvaluationType']
                    })
            
            if year_combo_data:
                yc_df = pd.DataFrame(year_combo_data)
                yc_counts = yc_df.groupby(['Year', 'Combination']).size().reset_index(name='Count')
                
                fig_temporal = create_plot(yc_counts, "Line Chart", 'Year', 'Count', 'Combination',
                                         "Evaluation Combinations Over Time", False, 400)
                st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Task-combination analysis
        st.subheader("📊 Tasks vs Evaluation Combinations")
        
        task_combo_data = []
        for item in eval_data:
            if item['EvaluationType'] != 'None' and item['Tasks']:
                for task in item['Tasks'][:2]:  # Limit to first 2 tasks
                    if task:
                        task_combo_data.append({
                            'Task': task,
                            'Combination': item['EvaluationType']
                        })
        
        if task_combo_data:
            tc_df = pd.DataFrame(task_combo_data)
            task_counts = tc_df['Task'].value_counts()
            top_tasks = task_counts.head(10).index
            
            filtered_tc = tc_df[tc_df['Task'].isin(top_tasks)]
            tc_counts = filtered_tc.groupby(['Task', 'Combination']).size().reset_index(name='Count')
            
            if not tc_counts.empty:
                fig_task_combo = create_advanced_heatmap(
                    tc_counts, 'Combination', 'Task', 'Count',
                    "Evaluation Combinations by NLG Task", 500
                )
                st.plotly_chart(fig_task_combo, use_container_width=True)

def show_criteria_comparison_analysis(eval_data, plot_type, show_advanced):
    st.subheader("📋 LLM vs Human Criteria Comparison")
    
    # Filter papers with both LLM and human evaluation
    llm_human_papers = [item for item in eval_data if item['LLM'] and item['Human']]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers", f"{len(eval_data):,}")
    with col2:
        st.metric("LLM + Human Papers", f"{len(llm_human_papers):,}")
    with col3:
        if len(eval_data) > 0:
            percentage = (len(llm_human_papers) / len(eval_data) * 100)
            st.metric("Overlap Percentage", f"{percentage:.1f}%")
    
    if not llm_human_papers:
        st.warning("No papers found with both LLM and human evaluation.")
        return
    
    # Collect criteria from papers with both evaluations
    llm_criteria = []
    human_criteria = []
    common_papers_criteria = []
    
    for item in llm_human_papers:
        llm_crits = item['LLMCriteria']
        human_crits = item['HumanCriteria']
        
        llm_criteria.extend(llm_crits)
        human_criteria.extend(human_crits)
        
        # Track criteria used in same paper
        for llm_crit in llm_crits:
            for human_crit in human_crits:
                common_papers_criteria.append({
                    'LLM_Criterion': llm_crit,
                    'Human_Criterion': human_crit,
                    'Conference': item['Conference'],
                    'Year': item['Year']
                })
    
    # Analyze criteria overlap
    llm_criteria_counts = Counter(llm_criteria)
    human_criteria_counts = Counter(human_criteria)
    
    llm_set = set(llm_criteria)
    human_set = set(human_criteria)
    common_criteria = llm_set.intersection(human_set)
    llm_only = llm_set - human_set
    human_only = human_set - llm_set
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Criteria Overlap Analysis")
        
        overlap_data = [
            {'Category': 'Common Criteria', 'Count': len(common_criteria)},
            {'Category': 'LLM-Only Criteria', 'Count': len(llm_only)},
            {'Category': 'Human-Only Criteria', 'Count': len(human_only)}
        ]
        
        overlap_df = pd.DataFrame(overlap_data)
        
        fig_overlap = px.pie(overlap_df, values='Count', names='Category',
                           title="LLM vs Human Criteria Overlap")
        fig_overlap.update_layout(height=400)
        st.plotly_chart(fig_overlap, use_container_width=True)
    
    with col2:
        st.subheader("Criteria Usage Frequency")
        
        if common_criteria:
            common_comparison = []
            for criterion in list(common_criteria)[:10]:  # Top 10 common
                common_comparison.extend([
                    {'Criterion': criterion, 'Type': 'LLM', 'Count': llm_criteria_counts[criterion]},
                    {'Criterion': criterion, 'Type': 'Human', 'Count': human_criteria_counts[criterion]}
                ])
            
            if common_comparison:
                comp_df = pd.DataFrame(common_comparison)
                
                fig_comparison = px.bar(comp_df, x='Criterion', y='Count', color='Type',
                                      title="Common Criteria: LLM vs Human Usage",
                                      barmode='group')
                fig_comparison.update_layout(height=400, xaxis_tickangle=45)
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"Common Criteria ({len(common_criteria)})")
        if common_criteria:
            for criterion in sorted(list(common_criteria))[:10]:
                llm_count = llm_criteria_counts[criterion]
                human_count = human_criteria_counts[criterion]
                st.write(f"• **{criterion}**")
                st.write(f"  LLM: {llm_count}, Human: {human_count}")
        else:
            st.info("No common criteria found")
    
    with col2:
        st.subheader(f"LLM-Only Criteria ({len(llm_only)})")
        if llm_only:
            llm_only_sorted = sorted([(c, llm_criteria_counts[c]) for c in llm_only], 
                                   key=lambda x: x[1], reverse=True)
            for criterion, count in llm_only_sorted[:10]:
                st.write(f"• **{criterion}**: {count}")
        else:
            st.info("No LLM-only criteria")
    
    with col3:
        st.subheader(f"Human-Only Criteria ({len(human_only)})")
        if human_only:
            human_only_sorted = sorted([(c, human_criteria_counts[c]) for c in human_only], 
                                     key=lambda x: x[1], reverse=True)
            for criterion, count in human_only_sorted[:10]:
                st.write(f"• **{criterion}**: {count}")
        else:
            st.info("No human-only criteria")
    
    if show_advanced:
        st.subheader("🔬 Advanced Criteria Analysis")
        
        # Criteria co-occurrence in same papers
        if common_papers_criteria:
            st.subheader("📊 Criteria Co-occurrence in Same Papers")
            
            cpp_df = pd.DataFrame(common_papers_criteria)
            
            # Get top LLM and human criteria for cleaner viz
            top_llm_criteria = [c for c, _ in llm_criteria_counts.most_common(8)]
            top_human_criteria = [c for c, _ in human_criteria_counts.most_common(10)]
            
            filtered_cpp = cpp_df[
                (cpp_df['LLM_Criterion'].isin(top_llm_criteria)) &
                (cpp_df['Human_Criterion'].isin(top_human_criteria))
            ]
            
            if not filtered_cpp.empty:
                cpp_counts = filtered_cpp.groupby(['LLM_Criterion', 'Human_Criterion']).size().reset_index(name='Count')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Heatmap of co-occurrences
                    fig_cooccur = create_advanced_heatmap(
                        cpp_counts, 'Human_Criterion', 'LLM_Criterion', 'Count',
                        "LLM vs Human Criteria Co-occurrence", 500
                    )
                    st.plotly_chart(fig_cooccur, use_container_width=True)
                
                with col2:
                    # Network-style scatter plot
                    fig_network = px.scatter(
                        cpp_counts,
                        x='LLM_Criterion',
                        y='Human_Criterion',
                        size='Count',
                        color='Count',
                        title="Criteria Relationship Network",
                        color_continuous_scale='Viridis',
                        height=500
                    )
                    fig_network.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_network, use_container_width=True)

def show_multidimensional_cross_analysis(eval_data, show_advanced):
    st.subheader("🌐 Multi-Dimensional Cross-Evaluation Analysis")
    
    # Create evaluation matrix
    eval_matrix = []
    for item in eval_data:
        eval_matrix.append({
            'Conference': item['Conference'],
            'Year': item['Year'],
            'Automatic': 1 if item['Automatic'] else 0,
            'LLM': 1 if item['LLM'] else 0,
            'Human': 1 if item['Human'] else 0,
            'EvaluationType': item['EvaluationType'],
            'TaskCount': len(item['Tasks']),
            'MetricCount': len(item['AutoMetrics']),
            'LLMModelCount': len(item['LLMModels']),
            'HumanCriteriaCount': len(item['HumanCriteria'])
        })
    
    eval_df = pd.DataFrame(eval_matrix)
    
    # Multi-dimensional bubble chart
    st.subheader("📊 Multi-Dimensional Evaluation Landscape")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Conference × Year bubble chart with evaluation complexity
        conf_year_complexity = eval_df.groupby(['Conference', 'Year']).agg({
            'Automatic': 'sum',
            'LLM': 'sum', 
            'Human': 'sum',
            'MetricCount': 'mean',
            'LLMModelCount': 'mean',
            'HumanCriteriaCount': 'mean'
        }).reset_index()
        
        # Calculate evaluation complexity score
        conf_year_complexity['ComplexityScore'] = (
            conf_year_complexity['MetricCount'] + 
            conf_year_complexity['LLMModelCount'] + 
            conf_year_complexity['HumanCriteriaCount']
        )
        
        fig_bubble = px.scatter(
            conf_year_complexity,
            x='Year',
            y='Conference',
            size='ComplexityScore',
            color='ComplexityScore',
            title="Evaluation Complexity by Conference × Year",
            color_continuous_scale='Viridis',
            hover_data=['Automatic', 'LLM', 'Human']
        )
        fig_bubble.update_layout(height=400)
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    with col2:
        # Evaluation method adoption rates by conference
        adoption_rates = eval_df.groupby('Conference')[['Automatic', 'LLM', 'Human']].mean().reset_index()
        adoption_melted = pd.melt(adoption_rates, id_vars=['Conference'], 
                                value_vars=['Automatic', 'LLM', 'Human'],
                                var_name='EvaluationMethod', value_name='AdoptionRate')
        
        fig_adoption = px.bar(adoption_melted, x='Conference', y='AdoptionRate', 
                            color='EvaluationMethod', barmode='group',
                            title="Evaluation Method Adoption by Conference")
        fig_adoption.update_layout(height=400)
        st.plotly_chart(fig_adoption, use_container_width=True)
    
    if show_advanced:
        st.subheader("🔍 Advanced Multi-Dimensional Analysis")
        
        # Correlation matrix of evaluation characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Evaluation Characteristics Correlation")
            
            correlation_cols = ['Automatic', 'LLM', 'Human', 'TaskCount', 'MetricCount', 'LLMModelCount', 'HumanCriteriaCount']
            corr_data = eval_df[correlation_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title="Evaluation Characteristics Correlation Matrix",
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.subheader("Evaluation Complexity Distribution")
            
            # Box plot of complexity by evaluation type
            complexity_data = []
            for _, row in eval_df.iterrows():
                complexity_score = row['MetricCount'] + row['LLMModelCount'] + row['HumanCriteriaCount']
                if row['EvaluationType'] != 'None':
                    complexity_data.append({
                        'EvaluationType': row['EvaluationType'],
                        'ComplexityScore': complexity_score
                    })
            
            if complexity_data:
                complexity_df = pd.DataFrame(complexity_data)
                
                fig_complexity = px.box(complexity_df, x='EvaluationType', y='ComplexityScore',
                                      title="Evaluation Complexity by Type")
                fig_complexity.update_layout(height=500, xaxis_tickangle=45)
                st.plotly_chart(fig_complexity, use_container_width=True)
        
        # 3D scatter plot of evaluation space
        st.subheader("🎯 3D Evaluation Space")
        
        # Create 3D representation
        eval_3d_data = eval_df[eval_df['EvaluationType'] != 'None'].copy()
        
        if not eval_3d_data.empty:
            fig_3d = px.scatter_3d(
                eval_3d_data,
                x='MetricCount',
                y='LLMModelCount', 
                z='HumanCriteriaCount',
                color='EvaluationType',
                size='TaskCount',
                hover_data=['Conference', 'Year'],
                title="3D Evaluation Methodology Space"
            )
            fig_3d.update_layout(height=600)
            st.plotly_chart(fig_3d, use_container_width=True)

def show_task_based_cross_analysis(eval_data, plot_type, show_percentage, show_advanced):
    st.subheader("🎯🔀 Task-based Cross-Evaluation Analysis")
    st.markdown("Analyzing evaluation approaches across different NLG tasks")
    
    # Extract task-evaluation combinations
    task_eval_data = []
    for item in eval_data:
        tasks = item['Tasks']
        eval_type = item['EvaluationType']
        
        for task in tasks[:3]:  # Limit tasks per paper
            if task:
                task_eval_data.append({
                    'Task': task,
                    'EvaluationType': eval_type,
                    'Automatic': item['Automatic'],
                    'LLM': item['LLM'],
                    'Human': item['Human'],
                    'Conference': item['Conference'],
                    'Year': item['Year'],
                    'AutoMetrics': item['AutoMetrics'],
                    'LLMCriteria': item['LLMCriteria'],
                    'HumanCriteria': item['HumanCriteria']
                })
    
    if not task_eval_data:
        st.warning("No task-evaluation data found.")
        return
    
    te_df = pd.DataFrame(task_eval_data)
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Tasks", te_df['Task'].nunique())
    with col2:
        st.metric("Unique Evaluation Types", te_df['EvaluationType'].nunique())
    with col3:
        st.metric("Task-Evaluation Pairs", len(task_eval_data))
    
    # Task-Evaluation Type Distribution
    st.subheader("📊 Evaluation Methods by Task")
    
    # Get most common tasks
    top_tasks = te_df['Task'].value_counts().head(12).index
    top_task_data = te_df[te_df['Task'].isin(top_tasks)]
    
    if not top_task_data.empty:
        # Count evaluation types per task
        task_eval_counts = top_task_data.groupby(['Task', 'EvaluationType']).size().reset_index(name='Count')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if plot_type == "Heatmap":
                # Create heatmap of tasks vs evaluation types
                heatmap_pivot = task_eval_counts.pivot(index='Task', columns='EvaluationType', values='Count').fillna(0)
                
                fig_heatmap = px.imshow(heatmap_pivot,
                                      labels=dict(x="Evaluation Type", y="Task", color="Count"),
                                      title="Task-Evaluation Method Usage Heatmap",
                                      color_continuous_scale="Blues")
                fig_heatmap.update_layout(height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                # Create regular plot
                fig = create_plot(task_eval_counts, plot_type, 'Task', 'Count', 'EvaluationType',
                                "Evaluation Methods Distribution by Task", show_percentage, 500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Task-Evaluation Combinations")
            top_combinations = task_eval_counts.nlargest(10, 'Count')
            for i, row in top_combinations.iterrows():
                st.write(f"**{row['Task']}** ({row['EvaluationType']}): {row['Count']}")
    
    # Advanced task-based analysis
    if show_advanced:
        st.subheader("🔬 Advanced Task-based Cross-Evaluation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tasks by Evaluation Diversity")
            
            # Calculate evaluation diversity per task
            eval_diversity = te_df.groupby('Task')['EvaluationType'].nunique().reset_index()
            eval_diversity.columns = ['Task', 'Unique_Eval_Types']
            eval_diversity = eval_diversity.sort_values('Unique_Eval_Types', ascending=False).head(10)
            
            if not eval_diversity.empty:
                fig_div = px.bar(eval_diversity, x='Task', y='Unique_Eval_Types',
                               title="Tasks with Most Diverse Evaluation Methods",
                               text='Unique_Eval_Types')
                fig_div.update_layout(xaxis_tickangle=-45, height=400)
                fig_div.update_traces(textposition='outside')
                st.plotly_chart(fig_div, use_container_width=True)
        
        with col2:
            st.subheader("Multi-Evaluation Usage by Task")
            
            # Find tasks with all three evaluation types
            multi_eval_tasks = []
            for task in top_tasks:
                task_data = te_df[te_df['Task'] == task]
                has_auto = task_data['Automatic'].any()
                has_llm = task_data['LLM'].any()
                has_human = task_data['Human'].any()
                
                if has_auto and has_llm and has_human:
                    count = len(task_data[
                        task_data['Automatic'] & 
                        task_data['LLM'] & 
                        task_data['Human']
                    ])
                    if count > 0:
                        multi_eval_tasks.append({
                            'Task': task,
                            'Multi_Eval_Count': count
                        })
            
            if multi_eval_tasks:
                multi_df = pd.DataFrame(multi_eval_tasks)
                multi_df = multi_df.sort_values('Multi_Eval_Count', ascending=False)
                
                fig_multi = px.bar(multi_df, x='Task', y='Multi_Eval_Count',
                                 title="Tasks Using All Three Evaluation Methods",
                                 text='Multi_Eval_Count')
                fig_multi.update_layout(xaxis_tickangle=-45, height=400)
                fig_multi.update_traces(textposition='outside')
                st.plotly_chart(fig_multi, use_container_width=True)
        
        # Task-specific evaluation patterns
        st.subheader("🎯 Task-Specific Evaluation Patterns")
        
        # Select a task for detailed analysis
        selected_task = st.selectbox(
            "Select a task for detailed evaluation analysis:",
            options=top_tasks.tolist(),
            key="task_analysis_selector"
        )
        
        if selected_task:
            task_data = te_df[te_df['Task'] == selected_task]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader(f"Automatic Metrics for {selected_task}")
                auto_metrics = []
                for item in task_data[task_data['Automatic']]['AutoMetrics']:
                    auto_metrics.extend(item)
                if auto_metrics:
                    metric_counts = Counter(auto_metrics)
                    top_metrics = metric_counts.most_common(8)
                    for metric, count in top_metrics:
                        st.write(f"• **{metric}**: {count}")
                else:
                    st.write("No automatic metrics data")
            
            with col2:
                st.subheader(f"LLM Criteria for {selected_task}")
                llm_criteria = []
                for item in task_data[task_data['LLM']]['LLMCriteria']:
                    llm_criteria.extend(item)
                if llm_criteria:
                    criteria_counts = Counter(llm_criteria)
                    top_criteria = criteria_counts.most_common(6)
                    for criterion, count in top_criteria:
                        st.write(f"• **{criterion}**: {count}")
                else:
                    st.write("No LLM criteria data")
            
            with col3:
                st.subheader(f"Human Criteria for {selected_task}")
                human_criteria = []
                for item in task_data[task_data['Human']]['HumanCriteria']:
                    human_criteria.extend(item)
                if human_criteria:
                    h_criteria_counts = Counter(human_criteria)
                    top_h_criteria = h_criteria_counts.most_common(6)
                    for criterion, count in top_h_criteria:
                        st.write(f"• **{criterion}**: {count}")
                else:
                    st.write("No human criteria data")
        
        # Temporal evolution of task evaluation patterns
        st.subheader("📈 Task Evaluation Evolution Over Time")
        
        # Get top 5 tasks for temporal analysis
        top_5_tasks = top_tasks[:5]
        temporal_data = te_df[te_df['Task'].isin(top_5_tasks)]
        
        if not temporal_data.empty:
            # Create temporal trend analysis
            yearly_task_eval = temporal_data.groupby(['Year', 'Task', 'EvaluationType']).size().reset_index(name='Count')
            
            # Focus on "All Three" evaluation type for clarity
            all_three_data = yearly_task_eval[yearly_task_eval['EvaluationType'] == 'All Three']
            
            if not all_three_data.empty:
                fig_temporal = create_plot(all_three_data, "Line Chart", 'Year', 'Count', 'Task',
                                         "Tasks Using All Three Evaluation Methods Over Time", False, 400)
                st.plotly_chart(fig_temporal, use_container_width=True)
            else:
                st.info("Not enough 'All Three' evaluation data for temporal analysis.")

def show_temporal_cross_analysis(eval_data, show_advanced):
    st.subheader("📈 Temporal Cross-Evaluation Analysis")
    
    # Yearly trends in evaluation adoption
    eval_df = pd.DataFrame(eval_data)
    
    yearly_trends = eval_df.groupby('Year')[['Automatic', 'LLM', 'Human']].agg(['sum', 'mean']).reset_index()
    yearly_trends.columns = ['Year', 'Auto_Count', 'Auto_Rate', 'LLM_Count', 'LLM_Rate', 'Human_Count', 'Human_Rate']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evaluation Method Adoption Trends")
        
        # Create trend data for plotting
        trend_data = []
        for _, row in yearly_trends.iterrows():
            trend_data.extend([
                {'Year': row['Year'], 'Method': 'Automatic', 'Rate': row['Auto_Rate']},
                {'Year': row['Year'], 'Method': 'LLM', 'Rate': row['LLM_Rate']},
                {'Year': row['Year'], 'Method': 'Human', 'Rate': row['Human_Rate']}
            ])
        
        trend_df = pd.DataFrame(trend_data)
        
        fig_trends = px.line(trend_df, x='Year', y='Rate', color='Method',
                           title="Evaluation Method Adoption Rates Over Time",
                           markers=True)
        fig_trends.update_layout(height=400)
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with col2:
        st.subheader("Evaluation Combination Evolution")
        
        # Track evolution of combination types
        combo_evolution = eval_df[eval_df['EvaluationType'] != 'None'].groupby(['Year', 'EvaluationType']).size().reset_index(name='Count')
        
        fig_combo_evolution = px.line(combo_evolution, x='Year', y='Count', color='EvaluationType',
                                    title="Evolution of Evaluation Combinations",
                                    markers=True)
        fig_combo_evolution.update_layout(height=400)
        st.plotly_chart(fig_combo_evolution, use_container_width=True)
    
    if show_advanced:
        st.subheader("🔬 Advanced Temporal Analysis")
        
        # Conference-wise evolution
        conf_year_eval = eval_df.groupby(['Conference', 'Year'])[['Automatic', 'LLM', 'Human']].mean().reset_index()
        
        tab1, tab2, tab3 = st.tabs(["Automatic Evolution", "LLM Evolution", "Human Evolution"])
        
        with tab1:
            fig_auto = px.line(conf_year_eval, x='Year', y='Automatic', color='Conference',
                             title="Automatic Evaluation Adoption by Conference",
                             markers=True)
            fig_auto.update_layout(height=400)
            st.plotly_chart(fig_auto, use_container_width=True)
        
        with tab2:
            fig_llm = px.line(conf_year_eval, x='Year', y='LLM', color='Conference',
                            title="LLM Evaluation Adoption by Conference",
                            markers=True)
            fig_llm.update_layout(height=400)
            st.plotly_chart(fig_llm, use_container_width=True)
        
        with tab3:
            fig_human = px.line(conf_year_eval, x='Year', y='Human', color='Conference',
                              title="Human Evaluation Adoption by Conference",
                              markers=True)
            fig_human.update_layout(height=400)
            st.plotly_chart(fig_human, use_container_width=True)
        
        # Evaluation maturity analysis
        st.subheader("📊 Evaluation Maturity Analysis")
        
        # Calculate "maturity" as combination of multiple evaluation types
        eval_df['MaturityScore'] = eval_df['Automatic'] + eval_df['LLM'] + eval_df['Human']
        
        maturity_trends = eval_df.groupby(['Year', 'Conference'])['MaturityScore'].mean().reset_index()
        
        fig_maturity = px.line(maturity_trends, x='Year', y='MaturityScore', color='Conference',
                             title="Evaluation Maturity Score Over Time",
                             markers=True)
        fig_maturity.update_layout(height=400)
        st.plotly_chart(fig_maturity, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Temporal Summary")
        
        summary_stats = eval_df.groupby('Year').agg({
            'Automatic': ['sum', 'mean'],
            'LLM': ['sum', 'mean'],
            'Human': ['sum', 'mean'],
            'EvaluationType': lambda x: len([t for t in x if t != 'None'])
        }).round(3)
        
        summary_stats.columns = ['Auto_Count', 'Auto_Rate', 'LLM_Count', 'LLM_Rate', 
                               'Human_Count', 'Human_Rate', 'Papers_With_Eval']
        
        st.dataframe(summary_stats, use_container_width=True)

if __name__ == "__main__":
    main()