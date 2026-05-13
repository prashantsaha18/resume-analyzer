#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements-vercel.txt

echo "Collecting static files..."
python manage.py collectstatic --noinput --settings=resumeai.settings_vercel

echo "Running migrations..."
python manage.py migrate --settings=resumeai.settings_vercel

echo "Build complete!"