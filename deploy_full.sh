#!/bin/bash

# Complete Deployment Script for vutuanchien.dev
# Author: Vu Tuan Chien

echo "🚀 Complete Deployment for vutuanchien.dev"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Initialize Git if needed
if [ ! -d .git ]; then
    echo -e "${YELLOW}Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit for vutuanchien.dev"
fi

# Step 2: Setup GitHub repository
echo -e "${BLUE}Step 1: Setup GitHub Repository${NC}"
echo "1. Create a new repository on GitHub named 'banking-test-generator'"
echo "2. Make it public"
echo "3. Don't initialize with README (we already have one)"
echo ""
read -p "Press Enter when you've created the repository..."

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Add remote
git remote add origin https://github.com/$GITHUB_USERNAME/banking-test-generator.git 2>/dev/null || git remote set-url origin https://github.com/$GITHUB_USERNAME/banking-test-generator.git

# Push main branch
echo -e "${GREEN}Pushing to GitHub main branch...${NC}"
git branch -M main
git push -u origin main

# Step 3: Deploy Frontend to GitHub Pages
echo -e "${BLUE}Step 2: Deploying Frontend to GitHub Pages${NC}"

# Create gh-pages branch
git checkout --orphan gh-pages

# Remove all files
git rm -rf .

# Copy only web files
cp -r web/* .
cp web/CNAME .

# Commit and push
git add .
git commit -m "Deploy frontend to GitHub Pages"
git push origin gh-pages --force

# Switch back to main
git checkout main

echo -e "${GREEN}✅ Frontend deployed to GitHub Pages!${NC}"
echo "Your frontend will be available at: https://vutuanchien.dev (after DNS propagation)"

# Step 4: Deploy Backend to Heroku
echo -e "${BLUE}Step 3: Deploying Backend to Heroku${NC}"
echo ""
echo "Prerequisites:"
echo "1. Install Heroku CLI: brew install heroku/brew/heroku"
echo "2. Login to Heroku: heroku login"
echo ""
read -p "Press Enter when ready to deploy to Heroku..."

# Create Heroku app
heroku create vutuanchien-testgen --region us

# Add buildpacks
heroku buildpacks:add heroku/python

# Deploy to Heroku
git push heroku main

# Scale dyno
heroku ps:scale web=1

# Open app
heroku open

echo -e "${GREEN}✅ Backend deployed to Heroku!${NC}"
echo ""
echo "=========================================="
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
echo ""
echo "Your application URLs:"
echo "  Frontend: https://vutuanchien.dev"
echo "  Backend: https://vutuanchien-testgen.herokuapp.com"
echo "  API Docs: https://vutuanchien-testgen.herokuapp.com/docs"
echo ""
echo "Next steps:"
echo "1. Configure DNS at Name.com:"
echo "   - Add A records: 185.199.108.153, 185.199.109.153, 185.199.110.153, 185.199.111.153"
echo "   - Or CNAME record: $GITHUB_USERNAME.github.io"
echo "2. Wait for DNS propagation (5-30 minutes)"
echo "3. Enable HTTPS in GitHub Pages settings"
