"""Application entry point.

Run the development server with:

    python app.py
"""

import os

from dotenv import load_dotenv

from app import create_app


load_dotenv()
application = create_app()


if __name__ == "__main__":
    application.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
