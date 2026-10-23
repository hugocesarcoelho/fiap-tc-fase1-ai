FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY breast-cancer-knn-svm-random-forest.ipynb .
COPY data/ data/

EXPOSE 8888

# Abre o Jupyter Notebook em http://localhost:8888 (sem token, apenas para uso local)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--IdentityProvider.token=", "--ServerApp.password="]
