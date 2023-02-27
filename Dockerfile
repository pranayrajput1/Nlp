FROM python:3.7

WORKDIR /usr/src/app

# Copy requirements and install them first so that layer is not rebuilt every time we build.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy over and install source code from this package.
COPY resume_screening ./resume_screening
#COPY setup.py ./setup.py
RUN pip install --no-cache-dir .
