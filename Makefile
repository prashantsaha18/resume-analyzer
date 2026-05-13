.PHONY: help install dev-backend dev-frontend migrate train-ml test deploy-check

help:
	@echo ""
	@echo "  ResumeAI Pro — Commands"
	@echo "  ─────────────────────────────────────"
	@echo "  make install       Install all dependencies"
	@echo "  make dev-backend   Start Django dev server"
	@echo "  make dev-frontend  Start React dev server"
	@echo "  make migrate       Run Django migrations"
	@echo "  make train-ml      Train ML models (5000 samples)"
	@echo "  make train-quick   Train ML models (1000 samples, fast)"
	@echo "  make test          Run backend tests"
	@echo "  make deploy-check  Check deployment readiness"
	@echo ""

install:
	@echo "Installing backend deps..."
	cd backend && python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	@echo "Installing frontend deps..."
	cd frontend && npm install

dev-backend:
	cd backend && . venv/bin/activate && python manage.py runserver

dev-frontend:
	cd frontend && npm start

migrate:
	cd backend && . venv/bin/activate && python manage.py makemigrations && python manage.py migrate

train-ml:
	cd backend && . venv/bin/activate && python manage.py train_models --samples 5000

train-quick:
	cd backend && . venv/bin/activate && python manage.py train_models --quick

test:
	cd backend && . venv/bin/activate && python manage.py test --verbosity=2

superuser:
	cd backend && . venv/bin/activate && python manage.py createsuperuser

deploy-check:
	@echo "Checking deployment readiness..."
	@test -f backend/.env && echo "  backend/.env exists" || echo "  MISSING: backend/.env"
	@test -f backend/firebase-credentials.json && echo "  firebase-credentials.json exists" || echo "  MISSING: firebase-credentials.json"
	@test -f frontend/.env && echo "  frontend/.env exists" || echo "  MISSING: frontend/.env"
	@echo "Done. Fix any MISSING items before deploying."