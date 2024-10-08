FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--port", "8000", "--host","0.0.0.0"]