#!/bin/bash

# Deployment Script for Banking Test Generator
# Author: Vu Tuan Chien

echo "🚀 Banking Test Generator Deployment Script"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${YELLOW}Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit"
fi

# Get GitHub repo URL
read -p "Enter your GitHub repository URL: " GITHUB_URL
git remote add origin $GITHUB_URL 2>/dev/null || git remote set-url origin $GITHUB_URL

# Push to main branch
echo -e "${GREEN}Pushing to GitHub main branch...${NC}"
git push -u origin main

# Deploy frontend to GitHub Pages
echo -e "${GREEN}Setting up GitHub Pages deployment...${NC}"
git checkout -b gh-pages 2>/dev/null || git checkout gh-pages

# Copy web files to root for GitHub Pages
cp -r web/* .
git add .
git commit -m "Deploy frontend to GitHub Pages"
git push origin gh-pages

# Switch back to main
git checkout main

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Go to Railway.app and connect your GitHub repo"
echo "2. Update web/config.js with your Railway backend URL"
echo "3. Configure custom domain in GitHub Pages settings"
echo ""
echo "Your app will be available at:"
echo "  - Frontend: https://[your-username].github.io/[repo-name]"
echo "  - Backend: https://[your-app].railway.app"
