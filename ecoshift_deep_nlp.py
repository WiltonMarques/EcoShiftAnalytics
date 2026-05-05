import pandas as pd
import json
import torch
import numpy as np
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- CONFIGURAÇÃO VISUAL ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'

print(f"{CIANO}{'='*80}\n ECOSHIFT ANALYTICS // TRIAGEM NEURAL COM PERSISTÊNCIA EM BANCO V7.2\n{'='*80}{RESETAR}")

class DeepNLPEngine:
    def __init__(self):
        self.engine = self._conectar_banco()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modelo Otimizado para o Português (BERTimbau)
        self.model_name = 'neuralmind/bert-base-portuguese-cased'
        print(f"{AMARELO}[+] Carregando hardware: {self.device.type.upper()}{RESETAR}")
        print(f"{AMARELO}[+] Inicializando Rede Neural: {self.model_name}...{RESETAR}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _conectar_banco(self, arquivo_config="db_config.json"):
        try:
            with open(arquivo_config, 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
            return create_engine(db_uri)
        except Exception as e:
            print(f"{VERMELHO}[ERRO] Falha ao conectar no banco: {e}{RESETAR}")
            return None

    def gerar_embeddings(self, textos, batch_size=16):
        """ Converte texto em vetores contextuais densos """
        embeddings_lista = []
        for i in tqdm(range(0, len(textos), batch_size), desc="Processando Semântica", unit="lote"):
            lote = textos[i:i+batch_size]
            inputs = self.tokenizer(lote, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean Pooling para capturar a essência da ementa
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            lote_embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            embeddings_lista.append(lote_embeddings.cpu().numpy())
            
        return np.vstack(embeddings_lista)

    def processar_e_salvar(self):
        # 1. Extração
        print(f"{AMARELO}[+] Extraindo ementas da base bruta...{RESETAR}")
        query = 'SELECT id, "siglaTipo", numero, ano, ementa FROM tb_camara_proposicoes WHERE ementa IS NOT NULL'
        try:
            df = pd.read_sql(query, self.engine)
            df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        except Exception as e:
            print(f"{VERMELHO}[X] Erro na extração: {e}{RESETAR}")
            return
            
        # 2. Manifesto de Sustentabilidade (O alvo da busca neural)
        manifesto = "Projetos de sustentabilidade, proteção ambiental, transição energética e créditos de carbono."
        vetor_alvo = self.gerar_embeddings([manifesto], batch_size=1)
        
        # 3. Cálculo de Similaridade Neural
        matriz_leis = self.gerar_embeddings(df['ementa'].tolist(), batch_size=32)
        df['eco_score_neural'] = cosine_similarity(matriz_leis, vetor_alvo).flatten() * 100
        
        # 4. PERSISTÊNCIA NO POSTGRESQL
        print(f"\n{AMARELO}[+] Salvando resultados na tabela 'tb_ecoshift_deep_ranking'...{RESETAR}")
        df_ranking = df[['id', 'siglaTipo', 'numero', 'ano', 'eco_score_neural']].copy()
        
        try:
            df_ranking.to_sql('tb_ecoshift_deep_ranking', self.engine, if_exists='replace', index=False)
            print(f"{VERDE}[OK] Matriz de Inteligência salva com sucesso no banco de dados.{RESETAR}")
        except Exception as e:
            print(f"{VERMELHO}[ERRO] Falha ao persistir no banco: {e}{RESETAR}")

        # 5. Backup em CSV (Opcional)
        df_ranking.to_csv('dossie_ambiental_deep_learning.csv', index=False, sep=';', decimal=',')
        
        # 6. Exibição do Top 5 (CORRIGIDA)
        print(f"\n{CIANO}>>> TOP 5 PROJETOS DETECTADOS PELA REDE NEURAL <<<{RESETAR}")
        df_exibicao = df.copy()
        df_exibicao['Projeto'] = df_exibicao['siglaTipo'] + " " + df_exibicao['numero'].astype(str) + "/" + df_exibicao['ano'].astype(str)
        
        top_5 = df_exibicao.sort_values(by='eco_score_neural', ascending=False).head(5)
        for index, row in top_5.iterrows():
            print(f" {VERDE}Score: {row['eco_score_neural']:.1f}%{RESETAR} | {row['Projeto']}")
            print(f" Resumo: {row['ementa'][:120]}...\n")

if __name__ == "__main__":
    engine = DeepNLPEngine()
    if engine.engine:
        engine.processar_e_salvar()