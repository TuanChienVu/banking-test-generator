#!/bin/bash

echo "🚀 Deploying to GitHub Pages (vutuanchien.dev)..."

# Commit và push changes
echo "📦 Committing changes..."
git add -A
git commit -m "Deploy frontend-only version to GitHub Pages"
git push origin main

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📝 Next steps:"
echo "1. Go to GitHub repo settings"
echo "2. Navigate to Pages section"
echo "3. Set Source: Deploy from branch"
echo "4. Branch: main, folder: / (root)"
echo "5. Save changes"
echo ""
echo "🌐 DNS Configuration at Name.com:"
echo "Add these A records:"
echo "  - 185.199.108.153"
echo "  - 185.199.109.153"
echo "  - 185.199.110.153"
echo "  - 185.199.111.153"
echo ""
echo "Your site will be available at:"
echo "  - https://vutuanchien.dev"
echo "  - https://tuanchienvu.github.io/banking-test-generator"
echo ""
echo "Note: DNS propagation may take up to 24 hours"
