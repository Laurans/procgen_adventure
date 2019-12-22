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
COPY base_dependencies.txt /tmp/base_dependencies.txt
RUN apt-get update && \
    apt-get install -y $(cat /tmp/base_dependencies.txt) && \
    apt-get clean


RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chmod g+w /etc/passwd && \
    chmod +x /usr/local/bin/fix-permissions && \
    fix-permissions $HOME 

USER $NB_UID
WORKDIR $HOME

# Install pyenv
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PATH:$PYENV_ROOT/bin

RUN cd $HOME && curl https://pyenv.run | bash && \
    echo "export PATH=\"${PYENV_ROOT}/bin:\$PATH\"" >> .bashrc && \
    echo "eval \"\$(pyenv init -)\"" >> .bashrc && \
    echo "eval \"\$(pyenv virtualenv-init -)\"" >> .bashrc && \
    pyenv install 3.7.4 && pyenv global 3.7.4

# Install Poetry
COPY requirements.txt /tmp/requirements.txt
RUN eval "$(pyenv init -)" && pip install --upgrade pip setuptools && pip install -r /tmp/requirements.txt 
WORKDIR ${HOME}/workdir