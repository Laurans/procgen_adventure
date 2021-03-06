FROM nvidia/cuda:10.0-cudnn7-runtime

# Configure environment
ENV NB_USER=developer \
    NB_UID=1000 \
    NB_GID=1000 \
    DEBIAN_FRONTEND=noninteractive

ENV HOME=/home/$NB_USER

ADD fix-permissions /usr/local/bin/fix-permissions


# Install base system libraries
COPY ./ /tmp/
RUN apt-get update && \
    apt-get --no-install-recommends install -y $(cat /tmp/base_dependencies.txt) && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get --no-install-recommends install -y python3.7 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chmod g+w /etc/passwd && \
    chmod +x /usr/local/bin/fix-permissions && \
    fix-permissions $HOME

USER $NB_UID
WORKDIR $HOME

ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PATH:$PYENV_ROOT/bin

RUN cd $HOME && curl https://pyenv.run | bash && \
    echo "export PATH=\"${PYENV_ROOT}/bin:\$PATH\"" >> .bashrc && \
    echo "eval \"\$(pyenv init -)\"" >> .bashrc && \
    echo "eval \"\$(pyenv virtualenv-init -)\"" >> .bashrc && \
    pyenv install 3.7.6 && pyenv global 3.7.6

# Install dependencies
RUN eval "$(pyenv init -)" && pip install --upgrade pip setuptools torch && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python && \
    echo "export PATH=\"${HOME}/.poetry/bin:\$PATH\"" >> .bashrc && . $HOME/.poetry/env && poetry config virtualenvs.create false
