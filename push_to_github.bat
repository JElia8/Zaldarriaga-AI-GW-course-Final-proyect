@echo off
REM Run this script from inside the project folder.
REM It initialises git (if not already done), commits everything, and pushes to GitHub.
REM You will be prompted for your GitHub username and a Personal Access Token (PAT).

cd /d "%~dp0"

git init
git config user.email "jp.elia@df.uba.ar"
git config user.name "Elia"

git branch -M main

REM Add remote (safe to run even if it already exists)
git remote remove origin 2>nul
git remote add origin https://github.com/JElia8/Zaldarriaga-AI-GW-course-Final-proyect.git

git add -A
git commit -m "Initial commit: Zaldarriaga AI GW course final project"

echo.
echo Pushing to GitHub...  Enter your GitHub username and a Personal Access Token when prompted.
echo (Create a PAT at: https://github.com/settings/tokens  -- needs 'repo' scope)
echo.
git push -u origin main

pause
