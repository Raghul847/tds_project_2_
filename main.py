# main.py (or app.py)
import os
import tempfile
import logging
from flask import Flask, request, jsonify
import time
import traceback

# Import the core logic function
from data_analysis import process_analysis_task

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define timeout (3 minutes = 180 seconds)
TIMEOUT_SECONDS = 3 * 60

# --- Initialize Flask App ---
app = Flask(__name__)
# Set maximum file upload size (e.g., 1MB, adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 

# --- API Endpoint ---
@app.route('/api/', methods=['POST'])
def analyze_data():
    """
    API endpoint to receive a data analysis task description.
    Expects a file upload with the key 'file'.
    Example: curl -X POST -F "file=@question.txt" http://localhost:8080/api/
    """
    start_time = time.time()
    logger.info("Received POST request to /api/")

    # --- 1. Validate Request ---
    if 'file' not in request.files:
        logger.error("No 'file' part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        logger.error("No file selected for upload")
        return jsonify({"error": "No selected file"}), 400

    # --- 2. Save File Temporarily ---
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
            # Read the file content and write it to the temporary file
            file_content = file.read().decode('utf-8')
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        logger.info(f"Uploaded file saved to temporary path: {tmp_file_path}")

        # --- 3. Process the Task ---
        logger.info("Calling process_analysis_task...")
        result = process_analysis_task(tmp_file_path)
        logger.info("process_analysis_task completed successfully.")

        # --- 4. Check Timeout ---
        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT_SECONDS:
            logger.warning(f"Processing exceeded time limit of {TIMEOUT_SECONDS}s (took {elapsed_time:.2f}s)")
            return jsonify({"error": f"Processing exceeded time limit of {TIMEOUT_SECONDS} seconds"}), 408

        # --- 5. Return Result ---
        logger.info(f"Returning result. Request processed in {elapsed_time:.2f} seconds.")
        # jsonify automatically sets Content-Type to application/json
        return jsonify(result), 200

    except UnicodeDecodeError as e:
        logger.error(f"Uploaded file is not valid UTF-8 text: {e}")
        return jsonify({"error": "Uploaded file must be a UTF-8 encoded text file"}), 400

    except ValueError as e: # Catch specific errors from data_analysis.py
        logger.error(f"ValueError in data analysis: {e}")
        return jsonify({"error": f"Analysis Error: {str(e)}"}), 400 # Bad Request if task is unrecognized/invalid

    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        # Return a generic error message to the client
        return jsonify({"error": "Internal server error during analysis"}), 500

    finally:
        # --- 6. Cleanup Temporary File ---
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                logger.info(f"Temporary file {tmp_file_path} deleted.")
            except OSError as e:
                logger.warning(f"Could not delete temporary file {tmp_file_path}: {e}")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Get port from environment variable (useful for cloud deployments) or default to 8080
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    # Run the app. host='0.0.0.0' is important for Docker/cloud deployments.
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False in production
