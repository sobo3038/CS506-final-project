# Define the environment variables
APP_NAME = app.py
VENV_DIR = venv

# Create a virtual environment and install dependencies
install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Activating virtual environment and installing dependencies..."
	. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Run the Flask application
run:
	@echo "Activating virtual environment and running Flask app..."
	. $(VENV_DIR)/bin/activate && flask run --host=0.0.0.0 --port=5000

# Clean up virtual environment
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed!"

# Generate requirements.txt
freeze:
	@echo "Freezing dependencies to requirements.txt..."
	. $(VENV_DIR)/bin/activate && pip freeze > requirements.txt
	@echo "Requirements saved!"