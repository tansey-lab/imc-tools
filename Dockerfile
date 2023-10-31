FROM python:3.11

RUN apt-get update && apt-get install cmake libcgal-dev libeigen3-dev swig
RUN pip install --upgrade pip

RUN git clone https://github.com/cgal/cgal-swig-bindings && \
    cd cgal-swig-bindings && \
    mkdir -p build/CGAL-5.0_release && \
    cd build/CGAL-5.0_release && \
    mkdir /cgal_python && \
    cmake -DCGAL_DIR=/usr/include/CGAL -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DBUILD_PYTHON=ON -DBUILD_JAVA=OFF -DPYTHON_OUTDIR_PREFIX=/usr/local/lib/python3.11/site-packages -DCMAKE_BUILD_TYPE=Release ../..

RUN pip install ipython
RUN mkdir -p /app
COPY src/ /app/src
COPY setup.py /app
COPY pyproject.toml /app

RUN cd /app && \
    pip install .
