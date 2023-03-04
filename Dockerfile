FROM python:3.10 AS playlist-back

# Create an application directory
RUN mkdir -p /app

ENV PYTHONPATH="/app"
WORKDIR "${PYTHONPATH}"

RUN pip install --user --upgrade pip
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --user -r /requirements.txt

COPY data data
COPY src src
COPY web web
COPY nltk.txt nltk.txt
COPY wsgi.py wsgi.py

EXPOSE 8019

ENTRYPOINT ["python3"]
CMD ["wsgi.py"]





