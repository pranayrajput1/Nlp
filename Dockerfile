FROM python:3.7

WORKDIR /usr/src/app

# Copy requirements and install them first so that layer is not rebuilt every time we build.
COPY requirements ./requirements
RUN pip install --no-cache-dir -r requirements/install.txt

# Copy over and install source code from this package.
COPY resume_screening ./resume_screening
COPY README.md ./README.md
#COPY setup.py ./setup.py
RUN pip install --no-cache-dir .