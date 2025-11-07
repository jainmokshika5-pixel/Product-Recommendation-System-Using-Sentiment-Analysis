#!/bin/bash

# Instructions:
# 1. Create a new repository on GitHub (https://github.com/new)
# 2. Copy the repository URL
# 3. Run this script with: bash push_to_new_repo.sh YOUR_REPO_URL
#
# Example: bash push_to_new_repo.sh https://github.com/Manthan3006/ai-product-recommendation.git

if [ -z "$1" ]; then
    echo "❌ Error: Please provide the GitHub repository URL"
    echo "Usage: bash push_to_new_repo.sh <repository-url>"
    echo "Example: bash push_to_new_repo.sh https://github.com/username/repo-name.git"
    exit 1
fi

REPO_URL=$1

echo "🚀 Setting up new GitHub repository..."
echo "📍 Repository URL: $REPO_URL"

# Add the new remote
git remote add origin "$REPO_URL"

# Verify the remote
echo "✅ Remote added:"
git remote -v

# Push to the new repository
echo "📤 Pushing to GitHub..."
git push -u origin main

echo "✅ Successfully pushed to new repository!"
echo "🔗 Your repository: $REPO_URL"
