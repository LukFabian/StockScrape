# Python Web Application Installation Guide

## Prerequisites
Ensure you have the following installed on your system:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.x](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)

## Installation Steps

### 1. Clone the Repository
```sh
git clone https://github.com/LukFabian/StockScrape
cd StockScrape
```

### 2. Start Services with Docker Compose
```sh
docker compose up -d
```
This will start any necessary services such as the database

### 3. Install Python Dependencies
```sh
pip install -r requirements.txt
```
This ensures all required dependencies for the application are installed.

### 4. Run Database Migrations
```sh
alembic upgrade head
```
This will update your database to the newest version

### 5. Run the FastAPI Application
```sh
fastapi dev app/main.py
```
This will start the FastAPI development server.

## Accessing the Application
Once the application is running, you can access it in your browser at:
```
http://localhost:8000
```

To view the interactive API documentation, go to:
```
http://localhost:8000/docs
```

## Stopping the Application
To stop the Docker containers, run:
```sh
docker compose down
```
