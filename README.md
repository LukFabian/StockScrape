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

# Features

## SGP-LSTM Stock Return Prediction

### Hybrid SGP + LSTM Model
Combines Symbolic Genetic Programming (SGP) for feature generation with LSTM networks for time-series forecasting.  
Inspired by [this research paper](https://www.nature.com/articles/s41598-023-50783-0).

### Technical Indicator Support
Leverages ADX, DMI, and RSI to inform predictive features.

### Cross-Sectional Return Forecasting
Classifies future returns as above or below the median over a specified window.

### High Predictive Performance
Demonstrated accuracy improvements over traditional methods.

### Multi-Stock Scalability
Handles large volumes of stock data for broad market analysis.

### Frontend Integration
Outputs are easily visualized via connected Vue.js components.