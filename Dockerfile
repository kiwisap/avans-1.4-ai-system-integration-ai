# Pin to Debian 12 (bookworm): the Microsoft ODBC repo below is debian/12, and
# the generic :slim tag has since moved on to Debian 13 (trixie).
FROM python:3.14-slim-bookworm

# --- ODBC Driver 18 for SQL Server (needed for pyodbc / Azure SQL) --------
# The Microsoft prod.list references signed-by=/usr/share/keyrings/microsoft-prod.gpg,
# so the key MUST live there (dearmored to .gpg), otherwise apt rejects the repo
# as "not signed".
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg2 ca-certificates \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL https://packages.microsoft.com/config/debian/12/prod.list \
        -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, for better build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
