# 🚀 Deployment Guide

## Deploy to Streamlit Community Cloud (Recommended)

### Prerequisites
1. GitHub account
2. Streamlit account (free at [streamlit.io](https://streamlit.io))

### Steps

#### 1. Create GitHub Repository
```bash
# Navigate to your project directory
cd nlg-user-ui

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: NLG Paper Trends Analysis UI"

# Add remote origin (replace with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/nlg-paper-analysis.git

# Push to GitHub
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `YOUR_USERNAME/nlg-paper-analysis`
5. Set branch to `main`
6. Set main file path to `app.py`
7. Click "Deploy!"

#### 3. Access Your App
Your app will be available at:
`https://YOUR_USERNAME-nlg-paper-analysis-app-xyz123.streamlit.app`

## Alternative Deployment Options

### Hugging Face Spaces
1. Create account at [huggingface.co](https://huggingface.co)
2. Go to [Spaces](https://huggingface.co/spaces)
3. Click "Create new Space"
4. Choose "Streamlit" as SDK
5. Upload your files or connect GitHub repo

### Railway
1. Create account at [railway.app](https://railway.app)
2. Connect GitHub repo
3. Railway will auto-detect Streamlit app
4. Deploy with one click

### Render
1. Create account at [render.com](https://render.com)
2. Connect GitHub repo
3. Choose "Web Service"
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Important Notes

### Data File Size
- The `data/all_papers.json` file is large (~100MB+)
- GitHub has a 100MB file limit
- Consider using Git LFS (Large File Storage) for the data file:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/*.json"

# Add .gitattributes
git add .gitattributes

# Add and commit normally
git add data/all_papers.json
git commit -m "Add data file with LFS"
```

### Performance Optimization
- The app caches data automatically with `@st.cache_data`
- For very large datasets, consider data preprocessing
- Monitor memory usage in deployment logs

### Custom Domain (Optional)
Most platforms offer custom domain options:
- Streamlit Cloud: Custom domains in Team/Business plans
- Vercel/Netlify: Custom domains in free tier
- Railway/Render: Custom domains in paid plans

## Troubleshooting

### Common Issues
1. **Memory limit exceeded**: Optimize data loading or upgrade plan
2. **Build timeout**: Check requirements.txt for conflicting versions
3. **File not found errors**: Ensure all paths are relative to app.py

### Getting Help
- Streamlit Community: [discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: Create issues in your repository
- Documentation: [docs.streamlit.io](https://docs.streamlit.io)