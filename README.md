# 🌿 EcoShift Analytics | Inteligência Legislativa GovTech

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Data_Warehouse-blue?style=for-the-badge&logo=postgresql)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)

## 📌 Visão Geral
O **EcoShift Analytics** é um ecossistema autônomo de dados projetado para monitorar, classificar e prever o impacto socioambiental de Projetos de Lei (PL, PEC, PLP) em tramitação na Câmara dos Deputados. O sistema cruza **Processamento de Linguagem Natural (Deep Learning)** com **Inteligência Geoespacial (NER)** para gerar métricas de risco e governança ESG em tempo real.

## 🏗️ Arquitetura do Sistema
O pipeline foi construído com foco em resiliência e alta performance, dividido em 4 motores:

1. **Motor de Ingestão (`ingestao_camara.py`):** Coleta dados abertos utilizando sessões blindadas com algoritmo de *Exponential Backoff* para contornar instabilidades da API governamental.
2. **Motor Neural (`ecoshift_deep_nlp.py`):** Utiliza redes neurais Transformers (*BERTimbau / RoBERTaLexPT*) para análise semântica das ementas, gerando um *Eco-Score* de aderência climática.
3. **Motor Geográfico (`ecoshift_ner_geografico.py`):** Algoritmo NER (*spaCy*) que extrai localidades dos textos de lei e realiza *join* relacional com bases de embargos do IBAMA.
4. **Painel Executivo (`dashboard_ecoshift.py`):** Interface visual em Streamlit focada no cálculo de impacto GovTech (GGMA e GTMI).

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos e Instalação
Certifique-se de ter o PostgreSQL rodando localmente (porta 5432) com um banco de dados configurado no arquivo `db_config.json`. Em seguida, instale as dependências:

```bash
pip install pandas numpy sqlalchemy psycopg2-binary requests urllib3 scikit-learn torch transformers tqdm spacy streamlit plotly