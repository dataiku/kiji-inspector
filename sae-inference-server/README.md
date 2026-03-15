# SAE Feature Description Server

This project provides a FastAPI server for generating descriptions of features from a Sparse Autoencoder (SAE).

## Features

*   **FastAPI Server**: A lightweight and fast web server.
*   **Health Check**: A `/healthz` endpoint to monitor the server's status.
*   **Describe Endpoint**: A `/describe` endpoint that accepts a feature activation (a vector of floats) and returns a human-readable description.
*   **Batch Describe Endpoint**: A `/describe/batch` endpoint for processing multiple describe requests in a single call.
*   **Flexible Input**: The server can extract feature activations from different JSON request structures.

## API

### `/healthz`

*   **Method**: `GET`
*   **Description**: Returns the health status of the server.
*   **Success Response**: `{"status": "ok"}`

### `/describe`

*   **Method**: `POST`
*   **Description**: Describes a single feature activation.
*   **Request Body**: See `app/schemas.py` for the detailed request and response models. The server can accept `DescribeByActivationRequest` or `DescribeBySampleResponseRequest`.
*   **Success Response**: A JSON object containing the description.

### `/describe/batch`

*   **Method**: `POST`
*   **Description**: Describes a batch of feature activations.
*   **Request Body**: A JSON object containing a list of describe requests. See `app/schemas.py`.
*   **Success Response**: A JSON object containing a list of description results.

## Setup and Running

1.  **Create the virtual environment and install dependencies**:
    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    ```

2.  **Run the server**:
    ```bash
    python main.py
    ```

The server will be available at `http://localhost:8000`.
