# Ubuntu 18.04 base with Python runtime and pyodbc to connect to SQL Server
FROM ubuntu:18.04

WORKDIR /

# apt-get and system utilities
RUN apt-get update && apt-get install -y \
    curl apt-utils apt-transport-https debconf-utils gcc build-essential g++-5\
    && rm -rf /var/lib/apt/lists/*

# adding custom Microsoft repository
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

# install SQL Server drivers
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev

EXPOSE 80
EXPOSE 443

# install SQL Server tools
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y mssql-tools
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# python libraries
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

RUN apt install -y libjpeg-dev zlib1g-dev
RUN pip3 install --upgrade pip setuptools wheel

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update -y && apt-get install -y python3-tk

# install necessary locales, this prevents any locale errors related to Microsoft packages
RUN apt-get update && apt-get install -y locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

COPY ./requirements.txt ./requirements.txt

WORKDIR /

#RUN pip3 install -r /requirements.txt
RUN pip3 install flask
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install pyodbc

#RUN apt-get install --upgrade pip
#RUN pip3 install matplotlib==3.3.1
RUN pip3 install matplotlib
RUN pip3 install seaborn

#RUN pip3 install pyodbc SQLAlchemy
#Run pip3 install Flask Flask-SQLAlchemy

COPY . /

#ENTRYPOINT [ "python3" ]

EXPOSE 8000
CMD ["python3", "hello.py", "runserver", "0.0.0.0:8000"]