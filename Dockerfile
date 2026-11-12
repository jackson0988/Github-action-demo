FROM python:3.10-slim

# Run the app as a normal user, not root: safer on any server
RUN useradd -m -u 1000 user

WORKDIR /app

# Install packages first, so changing the code does not reinstall them
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY --chown=user src ./src
COPY --chown=user model ./model

USER user
ENV HOME=/home/user

# Render tells the app which port to use in $PORT. Everywhere else: 8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn serve:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
