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

def create_plot(df, plot_type, x_col, y_col, color_col=None, title="", use_percentage=False, height=400):
    """Create different types of plots based on user selection"""
    
    if use_percentage and 'Count' in df.columns:
        # Convert counts to percentages
        if color_col:
            # Group by color column and calculate percentages within each group
            df_pct = df.copy()
            df_pct['Percentage'] = df_pct.groupby(color_col)['Count'].transform(lambda x: x / x.sum() * 100)
            y_col = 'Percentage'
        else:
            df_pct = df.copy()
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
    
    else:  # Default to bar chart
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
    
    fig.update_layout(height=height)
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
        index=1
    )
    
    # Conference selection
    all_conferences = sorted(set(p['conference'] for p in data))
    selected_conferences = st.sidebar.multiselect(
        "Conferences:",
        options=all_conferences,
        default=all_conferences
    )
    
    # Year selection
    all_years = sorted(set(p['year'] for p in data))
    selected_years = st.sidebar.multiselect(
        "Years:",
        options=all_years,
        default=all_years
    )
    
    # Apply filters to get working dataset
    filtered_data = get_paper_subset(data, paper_subset, selected_conferences, selected_years)
    
    # Display filtered data statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Current Selection:**")
    st.sidebar.metric("Papers in Selection", f"{len(filtered_data):,}")
    if paper_subset != "All Papers":
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
    
    if analysis_type == "General Overview":
        show_general_overview(filtered_data, paper_subset)
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

def show_general_overview(data, paper_subset):
    st.header("🌟 General Overview")
    st.markdown(f"Analysis of **{paper_subset}** across conferences and years")
    
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
    
    # Plot controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", ["Bar Chart", "Line Chart", "Area Chart", "Stacked Bar"], key="overview_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="overview_pct")
    with col3:
        group_by = st.selectbox("Color By:", ["Conference", "Year"], key="overview_color")
    
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
            
            # Conference totals
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
        pivot_df = df.pivot_table(values='Count', index='Conference', columns='Year', fill_value=0)
        
        # Add totals
        pivot_df['Total'] = pivot_df.sum(axis=1)
        pivot_df.loc['Total'] = pivot_df.sum()
        
        st.dataframe(pivot_df, use_container_width=True)
    else:
        st.warning("No data available for the current selection.")

def show_nlg_overview(data):
    st.header("📖 NLG Papers Overview")
    st.markdown("Distribution of evaluation approaches in NLG papers across conferences and years")
    
    # Filter NLG papers
    nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes']
    
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
    
    # Evaluation approaches distribution
    eval_stats = defaultdict(lambda: defaultdict(int))
    
    for paper in nlg_papers:
        key = (paper['conference'], paper['year'])
        eval_type = get_evaluation_type(paper)
        eval_stats[key][eval_type] += 1
    
    # Prepare data for heatmap
    conferences = sorted(set(p['conference'] for p in nlg_papers))
    years = sorted(set(p['year'] for p in nlg_papers))
    eval_types = ['Automatic Only', 'LLM Only', 'Human Only', 'Automatic + LLM', 
                  'Automatic + Human', 'LLM + Human', 'All Three', 'None']
    
    # Create visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Evaluation Approaches Over Time")
        
        # Aggregate by year for trend analysis
        year_eval_stats = defaultdict(lambda: defaultdict(int))
        for paper in nlg_papers:
            eval_type = get_evaluation_type(paper)
            year_eval_stats[paper['year']][eval_type] += 1
        
        trend_data = []
        for year in years:
            for eval_type in eval_types:
                count = year_eval_stats[year][eval_type]
                if count > 0:
                    trend_data.append({
                        'Year': year,
                        'Evaluation Type': eval_type,
                        'Count': count
                    })
        
        trend_df = pd.DataFrame(trend_data)
        if not trend_df.empty:
            fig = px.line(trend_df, x='Year', y='Count', color='Evaluation Type',
                         title="Trends in Evaluation Approaches", markers=True)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Overall Distribution")
        
        # Overall evaluation type distribution
        overall_eval_counts = Counter()
        for paper in nlg_papers:
            eval_type = get_evaluation_type(paper)
            overall_eval_counts[eval_type] += 1
        
        eval_df = pd.DataFrame([
            {'Type': k, 'Count': v} for k, v in overall_eval_counts.items()
        ])
        
        if not eval_df.empty:
            fig = px.pie(eval_df, values='Count', names='Type',
                        title="Distribution of Evaluation Types")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def show_text_only_analysis(data):
    st.header("📝 Text-Only NLG Tasks Analysis")
    st.markdown("Analysis of NLG papers with text-only input tasks (excluding multimodal tasks)")
    
    # Filter NLG papers with text-only tasks
    nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes']
    text_only_papers = []
    
    for paper in nlg_papers:
        tasks = paper['answer_1'].get('tasks', [])
        if tasks and is_text_only_task(tasks):
            text_only_papers.append(paper)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total NLG Papers", f"{len(nlg_papers):,}")
    with col2:
        st.metric("Text-Only NLG Papers", f"{len(text_only_papers):,}")
    with col3:
        percentage = (len(text_only_papers) / len(nlg_papers) * 100) if nlg_papers else 0
        st.metric("Text-Only Percentage", f"{percentage:.1f}%")
    
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
    
    task_names = [task for task, count in top_tasks]
    task_counts = [count for task, count in top_tasks]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(x=task_counts, y=task_names, orientation='h',
                    title="Most Common Text-Only NLG Tasks")
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
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
            fig = px.bar(eval_df, x='Evaluation Type', y='Count',
                        title="Count by Evaluation Type")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def show_tasks_distribution(data):
    st.header("🎯 NLG Tasks Distribution Analysis")
    st.markdown("Distribution of NLG tasks across different conferences and years")
    
    nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes']
    
    # Collect all tasks
    all_tasks = []
    task_paper_map = defaultdict(list)
    
    for paper in nlg_papers:
        tasks = paper['answer_1'].get('tasks', [])
        for task in tasks:
            cleaned_task = clean_task_name(task)
            all_tasks.append({
                'task': cleaned_task,
                'conference': paper['conference'],
                'year': paper['year'],
                'paper': paper
            })
            task_paper_map[cleaned_task].append(paper)
    
    # Task popularity
    task_counts = Counter([item['task'] for item in all_tasks])
    top_20_tasks = task_counts.most_common(20)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 20 NLG Tasks")
        task_names = [task for task, count in top_20_tasks]
        task_counts_list = [count for task, count in top_20_tasks]
        
        fig = px.bar(x=task_counts_list, y=task_names, orientation='h',
                    title="Most Popular NLG Tasks")
        fig.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Task Statistics")
        total_task_instances = sum(task_counts.values())
        unique_tasks = len(task_counts)
        
        st.metric("Total Task Instances", f"{total_task_instances:,}")
        st.metric("Unique Tasks", f"{unique_tasks:,}")
        st.metric("Avg Tasks/Paper", f"{total_task_instances/len(nlg_papers):.1f}")
        
        st.subheader("Top Tasks Details")
        for i, (task, count) in enumerate(top_20_tasks[:10], 1):
            with st.expander(f"{i}. {task} ({count} papers)"):
                papers = task_paper_map[task][:3]  # Show first 3 papers
                for paper in papers:
                    st.write(f"• **{paper['conference']} {paper['year']}**: {paper['title'][:100]}...")
                if len(task_paper_map[task]) > 3:
                    st.write(f"... and {len(task_paper_map[task]) - 3} more papers")
    
    # Conference-Year-Task analysis
    st.subheader("Task Distribution by Conference and Year")
    
    # Select top tasks for detailed analysis
    selected_tasks = st.multiselect(
        "Select tasks to analyze:",
        options=[task for task, _ in top_20_tasks],
        default=[task for task, _ in top_20_tasks[:5]]
    )
    
    if selected_tasks:
        # Create heatmap data
        conference_year_task_data = []
        
        for item in all_tasks:
            if item['task'] in selected_tasks:
                conference_year_task_data.append(item)
        
        # Aggregate data
        heatmap_data = defaultdict(lambda: defaultdict(int))
        for item in conference_year_task_data:
            key = f"{item['conference']} {item['year']}"
            heatmap_data[key][item['task']] += 1
        
        # Convert to dataframe for visualization
        heatmap_df_data = []
        for conf_year in heatmap_data:
            for task in selected_tasks:
                count = heatmap_data[conf_year][task]
                heatmap_df_data.append({
                    'Conference-Year': conf_year,
                    'Task': task,
                    'Count': count
                })
        
        if heatmap_df_data:
            heatmap_df = pd.DataFrame(heatmap_df_data)
            
            # Create pivot table for heatmap
            pivot_df = heatmap_df.pivot(index='Task', columns='Conference-Year', values='Count')
            pivot_df = pivot_df.fillna(0)
            
            fig = px.imshow(pivot_df.values, 
                          x=pivot_df.columns, 
                          y=pivot_df.index,
                          aspect='auto',
                          title="Task Distribution Heatmap",
                          color_continuous_scale='Blues')
            fig.update_layout(height=max(400, len(selected_tasks) * 40))
            st.plotly_chart(fig, use_container_width=True)

def show_automatic_metrics_analysis(data, paper_subset):
    st.header("🤖 Automatic Metrics Analysis")
    st.markdown(f"Distribution of automatic evaluation metrics in **{paper_subset}**")
    
    # Filter papers with automatic metrics (data is already subset-filtered)
    auto_papers = [p for p in data if p['answer_2']['answer'] == 'Yes']
    
    # Plot controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        plot_type = st.selectbox("Plot Type:", ["Horizontal Bar", "Bar Chart", "Line Chart"], key="metrics_plot")
    with col2:
        show_percentage = st.checkbox("Show Percentages", value=False, key="metrics_pct")
    with col3:
        normalize_metrics = st.checkbox("Normalize Metric Names", value=True, key="metrics_normalize", 
                                      help="Merge similar metrics (e.g., BLEU-1, BLEU-2 → BLEU-N)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers in Subset", f"{len(data):,}")
    with col2:
        st.metric("Papers with Auto Metrics", f"{len(auto_papers):,}")
    with col3:
        percentage = (len(auto_papers) / len(data) * 100) if data else 0
        st.metric("Percentage with Auto Metrics", f"{percentage:.1f}%")
    
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
    
    # Metrics by conference and year
    st.subheader("Automatic Metrics Trends")
    
    # Select metrics for trend analysis
    selected_metrics = st.multiselect(
        "Select metrics to analyze trends:",
        options=[metric for metric, _ in top_metrics],
        default=[metric for metric, _ in top_metrics[:5]]
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
            
            # Count by year and metric
            yearly_counts = trend_df.groupby(['Year', 'Metric']).size().reset_index(name='Count')
            
            fig = px.line(yearly_counts, x='Year', y='Count', color='Metric',
                         title="Automatic Metrics Usage Over Time", markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Metrics by tasks
    st.subheader("Metrics by NLG Tasks")
    
    # Create metrics-tasks mapping
    metric_task_pairs = []
    for item in all_metrics:
        for task in item['tasks']:
            metric_task_pairs.append({
                'Metric': item['metric'],
                'Task': task,
                'Conference': item['conference'],
                'Year': item['year']
            })
    
    if metric_task_pairs and selected_metrics:
        pairs_df = pd.DataFrame(metric_task_pairs)
        filtered_pairs = pairs_df[pairs_df['Metric'].isin(selected_metrics)]
        
        # Get top tasks for selected metrics
        task_counts = filtered_pairs['Task'].value_counts().head(10)
        top_tasks_for_metrics = task_counts.index.tolist()
        
        # Filter for heatmap
        heatmap_pairs = filtered_pairs[filtered_pairs['Task'].isin(top_tasks_for_metrics)]
        
        if not heatmap_pairs.empty:
            # Create pivot table
            pivot_table = heatmap_pairs.groupby(['Task', 'Metric']).size().unstack(fill_value=0)
            
            fig = px.imshow(pivot_table.values,
                          x=pivot_table.columns,
                          y=pivot_table.index,
                          aspect='auto',
                          title="Metrics Usage by Task",
                          color_continuous_scale='Blues')
            fig.update_layout(height=max(400, len(top_tasks_for_metrics) * 30))
            st.plotly_chart(fig, use_container_width=True)

def show_llm_evaluation_analysis(data):
    st.header("🧠 LLM Evaluation Analysis")
    st.markdown("Distribution of LLM evaluation methods, models, and criteria")
    
    # Filter papers with LLM evaluation
    llm_papers = [p for p in data if p['answer_3']['answer'] == 'Yes']
    nlg_llm_papers = [p for p in llm_papers if p['answer_1']['answer'] == 'Yes']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers with LLM Eval", f"{len(llm_papers):,}")
    with col2:
        st.metric("NLG Papers with LLM Eval", f"{len(nlg_llm_papers):,}")
    with col3:
        percentage = (len(nlg_llm_papers) / len(llm_papers) * 100) if llm_papers else 0
        st.metric("NLG Percentage", f"{percentage:.1f}%")
    
    # Collect LLM evaluation data
    all_models = []
    all_methods = []
    all_criteria = []
    
    for paper in nlg_llm_papers:
        answer3 = paper['answer_3']
        
        # Models
        for model in answer3.get('models', []):
            all_models.append({
                'model': model,
                'conference': paper['conference'],
                'year': paper['year'],
                'tasks': [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
            })
        
        # Methods
        for method in answer3.get('methods', []):
            all_methods.append({
                'method': method,
                'conference': paper['conference'],
                'year': paper['year'],
                'tasks': [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
            })
        
        # Criteria
        for criterion in answer3.get('criteria', []):
            all_criteria.append({
                'criterion': criterion,
                'conference': paper['conference'],
                'year': paper['year'],
                'tasks': [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
            })
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Models", "Methods", "Criteria"])
    
    with tab1:
        st.subheader("LLM Models Used for Evaluation")
        
        if all_models:
            model_counts = Counter([item['model'] for item in all_models])
            top_models = model_counts.most_common(15)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_names = [model for model, count in top_models]
                model_counts_list = [count for model, count in top_models]
                
                fig = px.bar(x=model_counts_list, y=model_names, orientation='h',
                           title="Most Used LLM Models for Evaluation")
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Unique Models", f"{len(model_counts):,}")
                st.metric("Total Model Uses", f"{len(all_models):,}")
                
                st.subheader("Top Models")
                for i, (model, count) in enumerate(top_models[:10], 1):
                    st.write(f"**{i}. {model}**: {count}")
            
            # Models trend over time
            if len(set(item['year'] for item in all_models)) > 1:
                st.subheader("Model Usage Trends")
                
                selected_models = st.multiselect(
                    "Select models to track:",
                    options=[model for model, _ in top_models[:10]],
                    default=[model for model, _ in top_models[:5]],
                    key="model_select"
                )
                
                if selected_models:
                    trend_data = []
                    for item in all_models:
                        if item['model'] in selected_models:
                            trend_data.append({
                                'Year': item['year'],
                                'Model': item['model']
                            })
                    
                    if trend_data:
                        trend_df = pd.DataFrame(trend_data)
                        yearly_counts = trend_df.groupby(['Year', 'Model']).size().reset_index(name='Count')
                        
                        fig = px.line(yearly_counts, x='Year', y='Count', color='Model',
                                     title="LLM Model Usage Over Time", markers=True)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No LLM model data available.")
    
    with tab2:
        st.subheader("LLM Evaluation Methods")
        
        if all_methods:
            method_counts = Counter([item['method'] for item in all_methods])
            top_methods = method_counts.most_common(15)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                method_names = [method for method, count in top_methods]
                method_counts_list = [count for method, count in top_methods]
                
                fig = px.bar(x=method_counts_list, y=method_names, orientation='h',
                           title="Most Used LLM Evaluation Methods")
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Unique Methods", f"{len(method_counts):,}")
                st.metric("Total Method Uses", f"{len(all_methods):,}")
                
                st.subheader("Top Methods")
                for i, (method, count) in enumerate(top_methods[:10], 1):
                    st.write(f"**{i}. {method}**: {count}")
        else:
            st.info("No LLM method data available.")
    
    with tab3:
        st.subheader("LLM Evaluation Criteria")
        
        if all_criteria:
            criteria_counts = Counter([item['criterion'] for item in all_criteria])
            top_criteria = criteria_counts.most_common(15)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                criteria_names = [criterion for criterion, count in top_criteria]
                criteria_counts_list = [count for criterion, count in top_criteria]
                
                fig = px.bar(x=criteria_counts_list, y=criteria_names, orientation='h',
                           title="Most Used LLM Evaluation Criteria")
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Unique Criteria", f"{len(criteria_counts):,}")
                st.metric("Total Criteria Uses", f"{len(all_criteria):,}")
                
                st.subheader("Top Criteria")
                for i, (criterion, count) in enumerate(top_criteria[:10], 1):
                    st.write(f"**{i}. {criterion}**: {count}")
        else:
            st.info("No LLM criteria data available.")

def show_human_evaluation_analysis(data):
    st.header("👥 Human Evaluation Analysis")
    st.markdown("Distribution of human evaluation guidelines and criteria")
    
    # Filter papers with human evaluation
    human_papers = [p for p in data if p['answer_4']['answer'] == 'Yes']
    nlg_human_papers = [p for p in human_papers if p['answer_1']['answer'] == 'Yes']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers with Human Eval", f"{len(human_papers):,}")
    with col2:
        st.metric("NLG Papers with Human Eval", f"{len(nlg_human_papers):,}")
    with col3:
        percentage = (len(nlg_human_papers) / len(human_papers) * 100) if human_papers else 0
        st.metric("NLG Percentage", f"{percentage:.1f}%")
    
    # Collect human evaluation criteria
    all_criteria = []
    all_guidelines = []
    
    for paper in nlg_human_papers:
        answer4 = paper['answer_4']
        
        # Criteria
        for criterion in answer4.get('criteria', []):
            all_criteria.append({
                'criterion': criterion,
                'conference': paper['conference'],
                'year': paper['year'],
                'tasks': [clean_task_name(t) for t in paper['answer_1'].get('tasks', [])]
            })
        
        # Guidelines (collect unique guidelines)
        guideline = answer4.get('guideline', '').strip()
        if guideline:
            all_guidelines.append({
                'guideline': guideline,
                'conference': paper['conference'],
                'year': paper['year']
            })
    
    # Criteria analysis
    st.subheader("Human Evaluation Criteria")
    
    if all_criteria:
        criteria_counts = Counter([item['criterion'] for item in all_criteria])
        top_criteria = criteria_counts.most_common(20)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            criteria_names = [criterion for criterion, count in top_criteria]
            criteria_counts_list = [count for criterion, count in top_criteria]
            
            fig = px.bar(x=criteria_counts_list, y=criteria_names, orientation='h',
                       title="Most Used Human Evaluation Criteria")
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Unique Criteria", f"{len(criteria_counts):,}")
            st.metric("Total Criteria Uses", f"{len(all_criteria):,}")
            
            st.subheader("Top Criteria")
            for i, (criterion, count) in enumerate(top_criteria[:15], 1):
                st.write(f"**{i}. {criterion}**: {count}")
        
        # Criteria trends over time
        if len(set(item['year'] for item in all_criteria)) > 1:
            st.subheader("Criteria Usage Trends")
            
            selected_criteria = st.multiselect(
                "Select criteria to track:",
                options=[criterion for criterion, _ in top_criteria[:10]],
                default=[criterion for criterion, _ in top_criteria[:5]],
                key="criteria_select"
            )
            
            if selected_criteria:
                trend_data = []
                for item in all_criteria:
                    if item['criterion'] in selected_criteria:
                        trend_data.append({
                            'Year': item['year'],
                            'Criterion': item['criterion']
                        })
                
                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    yearly_counts = trend_df.groupby(['Year', 'Criterion']).size().reset_index(name='Count')
                    
                    fig = px.line(yearly_counts, x='Year', y='Count', color='Criterion',
                                 title="Human Evaluation Criteria Over Time", markers=True)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No human evaluation criteria data available.")
    
    # Guidelines analysis
    st.subheader("Human Evaluation Guidelines")
    
    if all_guidelines:
        st.metric("Papers with Guidelines", f"{len(all_guidelines):,}")
        
        # Show sample guidelines
        st.subheader("Sample Guidelines")
        
        # Group guidelines by conference and year for better organization
        guidelines_by_conf_year = defaultdict(list)
        for item in all_guidelines:
            key = f"{item['conference']} {item['year']}"
            guidelines_by_conf_year[key].append(item['guideline'])
        
        # Display guidelines
        sample_count = 0
        for conf_year, guidelines in sorted(guidelines_by_conf_year.items())[:10]:
            if sample_count >= 5:  # Limit to 5 samples
                break
            
            unique_guidelines = list(set(guidelines))[:2]  # Show up to 2 unique guidelines per conf-year
            
            for guideline in unique_guidelines:
                if sample_count >= 5:
                    break
                
                with st.expander(f"{conf_year} - Human Evaluation Guideline"):
                    st.write(guideline[:500] + ("..." if len(guideline) > 500 else ""))
                
                sample_count += 1
    else:
        st.info("No human evaluation guidelines data available.")

def show_cross_evaluation_analysis(data):
    st.header("🔀 Cross-Evaluation Comparison")
    st.markdown("Analysis of papers using multiple evaluation types and comparison of criteria")
    
    nlg_papers = [p for p in data if p['answer_1']['answer'] == 'Yes']
    
    # Papers with all three evaluation types
    triple_eval_papers = []
    for paper in nlg_papers:
        if (paper['answer_2']['answer'] == 'Yes' and 
            paper['answer_3']['answer'] == 'Yes' and 
            paper['answer_4']['answer'] == 'Yes'):
            triple_eval_papers.append(paper)
    
    st.subheader("Papers with All Three Evaluation Types")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total NLG Papers", f"{len(nlg_papers):,}")
    with col2:
        st.metric("Papers with All 3 Evals", f"{len(triple_eval_papers):,}")
    with col3:
        percentage = (len(triple_eval_papers) / len(nlg_papers) * 100) if nlg_papers else 0
        st.metric("Percentage", f"{percentage:.1f}%")
    
    if triple_eval_papers:
        # Show distribution by conference and year
        conf_year_counts = defaultdict(int)
        for paper in triple_eval_papers:
            key = f"{paper['conference']} {paper['year']}"
            conf_year_counts[key] += 1
        
        if conf_year_counts:
            conf_year_df = pd.DataFrame([
                {'Conference-Year': k, 'Count': v}
                for k, v in conf_year_counts.items()
            ])
            
            fig = px.bar(conf_year_df, x='Conference-Year', y='Count',
                        title="Papers with All Three Evaluation Types by Conference-Year")
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show sample papers
        st.subheader("Sample Papers with All Three Evaluation Types")
        
        for i, paper in enumerate(triple_eval_papers[:5], 1):
            with st.expander(f"{i}. {paper['title']} ({paper['conference']} {paper['year']})"):
                st.write(f"**Abstract**: {paper['abstract'][:300]}...")
                
                # Show evaluation details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Automatic Metrics:**")
                    for metric in paper['answer_2'].get('automatic_metrics', [])[:5]:
                        st.write(f"• {metric}")
                
                with col2:
                    st.write("**LLM Evaluation:**")
                    st.write(f"Models: {', '.join(paper['answer_3'].get('models', [])[:3])}")
                    st.write(f"Criteria: {', '.join(paper['answer_3'].get('criteria', [])[:3])}")
                
                with col3:
                    st.write("**Human Evaluation:**")
                    for criterion in paper['answer_4'].get('criteria', [])[:5]:
                        st.write(f"• {criterion}")
    
    # Comparison of criteria between LLM and Human evaluation
    st.subheader("LLM vs Human Evaluation Criteria Comparison")
    
    # Collect criteria from papers that have both LLM and human evaluation
    llm_human_papers = []
    for paper in nlg_papers:
        if (paper['answer_3']['answer'] == 'Yes' and 
            paper['answer_4']['answer'] == 'Yes'):
            llm_human_papers.append(paper)
    
    if llm_human_papers:
        st.metric("Papers with Both LLM & Human Eval", f"{len(llm_human_papers):,}")
        
        # Collect criteria
        llm_criteria = []
        human_criteria = []
        
        for paper in llm_human_papers:
            llm_criteria.extend(paper['answer_3'].get('criteria', []))
            human_criteria.extend(paper['answer_4'].get('criteria', []))
        
        # Count criteria
        llm_criteria_counts = Counter(llm_criteria)
        human_criteria_counts = Counter(human_criteria)
        
        # Find common and unique criteria
        llm_set = set(llm_criteria)
        human_set = set(human_criteria)
        
        common_criteria = llm_set.intersection(human_set)
        llm_only = llm_set - human_set
        human_only = human_set - llm_set
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Common Criteria", f"{len(common_criteria)}")
            if common_criteria:
                st.write("**Common Criteria:**")
                for criterion in sorted(list(common_criteria))[:10]:
                    llm_count = llm_criteria_counts[criterion]
                    human_count = human_criteria_counts[criterion]
                    st.write(f"• {criterion} (LLM: {llm_count}, Human: {human_count})")
        
        with col2:
            st.metric("LLM-Only Criteria", f"{len(llm_only)}")
            if llm_only:
                st.write("**LLM-Only Criteria:**")
                llm_only_counts = [(c, llm_criteria_counts[c]) for c in llm_only]
                llm_only_counts.sort(key=lambda x: x[1], reverse=True)
                for criterion, count in llm_only_counts[:10]:
                    st.write(f"• {criterion} ({count})")
        
        with col3:
            st.metric("Human-Only Criteria", f"{len(human_only)}")
            if human_only:
                st.write("**Human-Only Criteria:**")
                human_only_counts = [(c, human_criteria_counts[c]) for c in human_only]
                human_only_counts.sort(key=lambda x: x[1], reverse=True)
                for criterion, count in human_only_counts[:10]:
                    st.write(f"• {criterion} ({count})")
        
        # Visualization of criteria comparison
        if common_criteria:
            st.subheader("Criteria Usage Comparison")
            
            comparison_data = []
            for criterion in list(common_criteria)[:10]:  # Top 10 common criteria
                comparison_data.extend([
                    {'Criterion': criterion, 'Type': 'LLM', 'Count': llm_criteria_counts[criterion]},
                    {'Criterion': criterion, 'Type': 'Human', 'Count': human_criteria_counts[criterion]}
                ])
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                fig = px.bar(comparison_df, x='Criterion', y='Count', color='Type',
                           title="LLM vs Human Criteria Usage (Common Criteria)",
                           barmode='group')
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No papers found with both LLM and human evaluation.")

if __name__ == "__main__":
    main()