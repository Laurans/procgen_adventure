FROM nvidia/cuda:10.0-cudnn7-runtime

ARG NB_USER=developer
ARG NB_UID=1000
ARG NB_GID=1000


# Configure environment
ENV NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    DEBIAN_FRONTEND=noninteractive 

ENV HOME=/home/$NB_USER 

ADD fix-permissions /usr/local/bin/fix-permissions


# Install base system libraries
COPY ./* /tmp/
RUN apt-get update && \
    apt-get --no-install-recommends install -y $(cat /tmp/base_dependencies.txt) && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get --no-install-recommends install -y python3.7 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -s $(which python3.7) python


RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chmod g+w /etc/passwd && \
    chmod +x /usr/local/bin/fix-permissions && \
    fix-permissions $HOME 

USER $NB_UID
WORKDIR $HOME

# Install dependences 
#RUN pip install --upgrade pip setuptools && pip install -r /tmp/requirements.txt 
WORKDIR ${HOME}/workdir