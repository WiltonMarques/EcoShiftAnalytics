import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import json
import pandas as pd
from sqlalchemy import create_engine

# --- CONFIGURAÇÃO VISUAL ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'

print(f"{CIANO}{'='*80}\n ECOSHIFT ANALYTICS // PIPELINE DE INGESTÃO (EXPONENTIAL BACKOFF) V6.0\n{'='*80}{RESETAR}")

class CamaraConnector:
    """ Extrator de Dados Abertos e Loader para PostgreSQL com Resiliência Automática """
    
    def __init__(self, arquivo_config="db_config.json"):
        self.base_url = "https://dadosabertos.camara.leg.br/api/v2"
        self.engine = self._conectar_banco(arquivo_config)
        self.sessao_blindada = self._criar_sessao_resiliente()

    def _conectar_banco(self, arquivo_config):
        try:
            with open(arquivo_config, 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
            engine = create_engine(db_uri)
            print(f"{VERDE}[+] Conexão ativada com o Data Warehouse: {cfg['dbname']}{RESETAR}")
            return engine
        except Exception as e:
            print(f"{VERMELHO}[X] Falha na conexão com o banco: {e}{RESETAR}")
            return None

    def _criar_sessao_resiliente(self):
        """ Configura a estratégia de Exponential Backoff via HTTPAdapter """
        session = requests.Session()
        retries = Retry(
            total=5,                # Máximo de 5 tentativas por requisição
            backoff_factor=2,       # Fator multiplicador (0s, 2s, 4s, 8s, 16s)
            status_forcelist=[429, 500, 502, 503, 504], # Gatilhos para a pausa
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _salvar_no_banco(self, df, nome_tabela):
        if self.engine is None or df.empty:
            return
        try:
            for col in df.columns:
                if df[col].apply(type).eq(list).any() or df[col].apply(type).eq(dict).any():
                    df[col] = df[col].astype(str)
            df.to_sql(nome_tabela, self.engine, if_exists='append', index=False)
        except Exception:
            pass # Ignora duplicatas silenciosamente

    def extrair_proposicoes(self, max_paginas=100):
        print(f"\n{AMARELO}[+] Iniciando Varredura Histórica (2022-2026) com Backoff Ativado...{RESETAR}")
        proposicoes_coletadas = []
        pagina = 1
        
        while pagina <= max_paginas:
            url = f"{self.base_url}/proposicoes"
            params = {
                "pagina": pagina, 
                "itens": 100, 
                "ordem": "DESC", 
                "ordenarPor": "ano",
                "siglaTipo": ["PL", "PEC", "PLP"], 
                "ano": [2022, 2023, 2024, 2025, 2026] 
            }
            
            try:
                # A sessão gerencia as falhas e esperas internamente
                response = self.sessao_blindada.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    dados = response.json().get("dados", [])
                    if not dados:
                        print(f"{CIANO}[i] Fim dos registros encontrados.{RESETAR}")
                        break
                        
                    proposicoes_coletadas.extend(dados)
                    print(f"  -> Página {pagina}/{max_paginas} processada: {len(dados)} proposições capturadas.")
                    pagina += 1
                    time.sleep(0.3) 
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"{VERMELHO}[FALHA] API inacessível: {e}{RESETAR}")
                break
                
        if proposicoes_coletadas:
            df_proposicoes = pd.DataFrame(proposicoes_coletadas)
            self._salvar_no_banco(df_proposicoes, "tb_camara_proposicoes")
            print(f"{VERDE}[OK] {len(df_proposicoes)} projetos salvos com sucesso.{RESETAR}")
            return df_proposicoes
        return None

    def extrair_temas(self, id_proposicao):
        url = f"{self.base_url}/proposicoes/{id_proposicao}/temas"
        try:
            response = self.sessao_blindada.get(url, timeout=10)
            if response.status_code == 200:
                dados = response.json().get("dados", [])
                if dados:
                    df_temas = pd.DataFrame(dados)
                    df_temas['id_proposicao'] = id_proposicao
                    self._salvar_no_banco(df_temas, "tb_camara_temas")
                    return df_temas
        except Exception:
            pass
        return None

if __name__ == "__main__":
    extrator = CamaraConnector()
    if extrator.engine:
        df_props = extrator.extrair_proposicoes(max_paginas=100)
        if df_props is not None and not df_props.empty:
            print(f"\n{AMARELO}[+] Enriquecendo Temas (Resiliência Ativa)...{RESETAR}")
            for index, row in df_props.iterrows():
                extrator.extrair_temas(row['id'])
                time.sleep(0.2)
        print(f"\n{CIANO}[+] Pipeline concluído. O Data Lake está pronto para a análise de impacto.{RESETAR}")