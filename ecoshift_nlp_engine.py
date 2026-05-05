import pandas as pd
import json
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÃO VISUAL ECOSHIFT ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'

print(f"{CIANO}{'='*80}\n ECOSHIFT ANALYTICS // MOTOR NLP COM PERSISTÊNCIA EM BANCO DE DADOS V6.0\n{'='*80}{RESETAR}")

class NLPEngine:
    def __init__(self):
        self._preparar_nltk()
        self.engine = self._conectar_banco()
        self.stop_words = set(stopwords.words('portuguese'))
        
        jargoes = {
            'altera', 'lei', 'dispoe', 'sobre', 'art', 'inciso', 'redacao', 'estabelece', 'providencias',
            'parecer', 'relator', 'relatora', 'comissao', 'requer', 'aprovacao', 'substitutivo', 'adotado',
            'exarado', 'deputado', 'deputada', 'nacional', 'realizacao', 'providencia', 'representante',
            'projeto', 'proposicao', 'institui', 'cria', 'acrescenta', 'paragrafo', 'federal', 'membro', 'mesa',
            'solicita', 'informacoes', 'ministerio', 'ministro', 'voto', 'votos', 'favoravel', 'contrario',
            'tramitacao', 'urgencia', 'apresentacao', 'ementa', 'texto', 'outras'
        }
        self.stop_words.update(jargoes)

    def _preparar_nltk(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

    def _conectar_banco(self, arquivo_config="db_config.json"):
        try:
            with open(arquivo_config, 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
            return create_engine(db_uri)
        except Exception as e:
            print(f"{VERMELHO}[ERRO] Falha ao conectar no banco para NLP: {e}{RESETAR}")
            return None

    def extrair_ementas(self):
        print(f"{AMARELO}[+] Fase 1: Ingestão de Texto Bruto (PL/PEC/PLP)...{RESETAR}")
        query = """
        SELECT id, "siglaTipo", numero, ano, ementa 
        FROM tb_camara_proposicoes 
        WHERE ementa IS NOT NULL 
        AND "siglaTipo" IN ('PL', 'PEC', 'PLP')
        """
        try:
            df = pd.read_sql(query, self.engine)
            df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
            print(f"{VERDE}[OK] {len(df)} projetos únicos extraídos do PostgreSQL.{RESETAR}")
            return df
        except Exception as e:
            print(f"{VERMELHO}[X] Erro na extração: {e}{RESETAR}")
            return None

    def limpar_texto(self, texto):
        if not isinstance(texto, str): return ""
        texto = texto.lower()
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
        texto = re.sub(r'[^a-z]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        tokens = [palavra for palavra in texto.split() if palavra not in self.stop_words and len(palavra) > 3]
        return " ".join(tokens)

    def processar_matriz_nlp(self, df, num_clusters=5):
        print(f"{AMARELO}[+] Fase 2 e 3: Higienização e Construção da Matriz (TF-IDF)...{RESETAR}")
        df['ementa_limpa'] = df['ementa'].apply(self.limpar_texto)
        vetorizador = TfidfVectorizer(max_features=1500, max_df=0.85, min_df=2)
        matriz_tfidf = vetorizador.fit_transform(df['ementa_limpa'])
        
        print(f"{AMARELO}[+] Fase 4: Clusterização Não Supervisionada (K-Means)...{RESETAR}")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20, max_iter=500)
        df['cluster'] = kmeans.fit_predict(matriz_tfidf)
        
        return df, vetorizador, matriz_tfidf

    def calcular_eco_score(self, df, vetorizador, matriz_tfidf):
        print(f"\n{AMARELO}[+] Fase 5: Aplicando Motor de Similaridade por Cosseno (Eco-Score)...{RESETAR}")
        
        texto_ouro = """
        meio ambiente sustentabilidade ecologica protecao ambiental mudancas climaticas 
        aquecimento global desmatamento amazonia bioma preservacao conservacao 
        economia verde energia renovavel creditos mercado de carbono transicao energetica 
        poluicao floresta biodiversidade residuos solidos transicao ecologica eolica solar
        """
        texto_ouro_limpo = self.limpar_texto(texto_ouro)
        vetor_ouro = vetorizador.transform([texto_ouro_limpo])
        
        scores = cosine_similarity(matriz_tfidf, vetor_ouro).flatten()
        df['eco_score'] = scores * 100 
        
        # --- FASE 6: PERSISTÊNCIA NO BANCO DE DADOS ---
        print(f"{AMARELO}[+] Fase 6: Gravando Rankings e Scores no PostgreSQL...{RESETAR}")
        
        # Preparamos o DataFrame para o Banco (removendo a ementa limpa para economizar espaço)
        df_db = df[['id', 'siglaTipo', 'numero', 'ano', 'cluster', 'eco_score']].copy()
        
        try:
            # Gravamos na tabela 'tb_ecoshift_ranking' (se existir, ele substitui; se não, ele cria)
            df_db.to_sql('tb_ecoshift_ranking', self.engine, if_exists='replace', index=False)
            print(f"{VERDE}[OK] Matriz de inteligência persistida na tabela 'tb_ecoshift_ranking'.{RESETAR}")
        except Exception as e:
            print(f"{VERMELHO}[X] Erro ao gravar no banco: {e}{RESETAR}")

        # Ordenação para exibição
        df_ranking = df.sort_values(by='eco_score', ascending=False)
        
        print(f"{CIANO}\n>>> TOP 5 PROJETOS: IMPACTO SOCIOAMBIENTAL (EXTRAÍDO DO BANCO) <<<{RESETAR}")
        top_5 = df_ranking.head(5)
        for index, row in top_5.iterrows():
            if row['eco_score'] > 0:
                print(f" {VERDE}Score: {row['eco_score']:.1f}%{RESETAR} | {row['siglaTipo']} {row['numero']}/{row['ano']}")
                print(f" Resumo: {row['ementa'][:120]}...\n")
        
        return df_ranking

if __name__ == "__main__":
    nlp = NLPEngine()
    if nlp.engine:
        df_leis = nlp.extrair_ementas()
        if df_leis is not None and not df_leis.empty:
            df_classificado, modelo_tfidf, matriz = nlp.processar_matriz_nlp(df_leis)
            df_final = nlp.calcular_eco_score(df_classificado, modelo_tfidf, matriz)
            print(f"{CIANO}[+] Ecossistema de Inteligência Legislativa Finalizado e Armazenado.{RESETAR}")