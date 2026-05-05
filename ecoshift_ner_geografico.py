import spacy
import pandas as pd
import json
from sqlalchemy import create_engine

# --- CONFIGURAÇÃO VISUAL ---
VERDE, VERMELHO, CIANO, AMARELO, RESETAR = '\033[92m', '\033[91m', '\033[96m', '\033[93m', '\033[0m'

print(f"{CIANO}{'='*80}\n ECOSHIFT ANALYTICS // NER GEOGRÁFICO E CRUZAMENTO IBAMA V8.1\n{'='*80}{RESETAR}")

class GeoEntityExtractor:
    def __init__(self):
        print(f"{AMARELO}[+] Inicializando motor NER (pt_core_news_lg)...{RESETAR}")
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except OSError:
            print(f"{VERMELHO}[ERRO] Modelo não encontrado. Rode no terminal: python -m spacy download pt_core_news_lg{RESETAR}")
            exit()
            
        self.engine = self._conectar_banco()
        
        # Dicionário simulando a base do IBAMA (Municípios prioritários para prevenção na Amazônia)
        self.lista_embargos_ibama = [
            "Altamira", "São Félix do Xingu", "Porto Velho", "Lábrea", 
            "Novo Progresso", "Apuí", "Colniza", "Itaituba", "Amazônia", "Tapajós"
        ]

    def _conectar_banco(self):
        try:
            with open("db_config.json", 'r') as f:
                cfg = json.load(f)
            db_uri = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
            return create_engine(db_uri)
        except Exception as e:
            print(f"{VERMELHO}[X] Falha na conexão com o banco: {e}{RESETAR}")
            return None

    def extrair_e_cruzar_entidades(self):
        print(f"\n{AMARELO}[+] Lendo matriz de inteligência do Data Warehouse...{RESETAR}")
        query = """
        SELECT r.id, r."siglaTipo", r.numero, r.ano, r.eco_score_neural, p.ementa
        FROM tb_ecoshift_deep_ranking r
        JOIN tb_camara_proposicoes p ON r.id = p.id
        WHERE r.eco_score_neural > 20.0
        """
        df = pd.read_sql(query, self.engine)
        
        # --- O FILTRO SANITÁRIO (ANTI-PRODUTO CARTESIANO) ---
        # Garante que a IA processe apenas uma cópia de cada projeto, ignorando duplicatas de ingestão
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        
        if df.empty:
            print(f"{VERMELHO}[X] Nenhum projeto encontrado. Certifique-se de ter rodado o ecoshift_deep_nlp.py antes.{RESETAR}")
            return

        print(f"{VERDE}[OK] {len(df)} projetos únicos carregados. Iniciando varredura semântica e cruzamento geoespacial...{RESETAR}")

        locais_encontrados = []
        orgaos_encontrados = []
        alertas_ibama = []

        # Iteração sobre as leis para extração NER
        for index, row in df.iterrows():
            texto_ementa = str(row['ementa'])
            doc = self.nlp(texto_ementa)
            
            # Extrai apenas Entidades do tipo Localidade (LOC) e Organização (ORG)
            locais = [ent.text for ent in doc.ents if ent.label_ == 'LOC']
            orgaos = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            
            # Limpeza e remoção de duplicatas na mesma lei
            locais_unicos = list(set(locais))
            orgaos_unicos = list(set(orgaos))
            
            locais_encontrados.append(", ".join(locais_unicos))
            orgaos_encontrados.append(", ".join(orgaos_unicos))
            
            # CRUZAMENTO COM IBAMA: Verifica se algum local extraído bate com a lista de risco
            risco_detectado = False
            for local in locais_unicos:
                if any(embargo.lower() in local.lower() for embargo in self.lista_embargos_ibama):
                    risco_detectado = True
                    break
            
            alertas_ibama.append(risco_detectado)

        # Enriquecendo o DataFrame
        df['entidades_locais'] = locais_encontrados
        df['entidades_orgaos'] = orgaos_encontrados
        df['alerta_critico_desmatamento'] = alertas_ibama
        
        # Penalização/Bonificação do Score: Se toca em área do IBAMA, sobe a relevância
        df.loc[df['alerta_critico_desmatamento'] == True, 'eco_score_neural'] += 15.0

        # --- PERSISTÊNCIA ---
        print(f"{AMARELO}[+] Salvando cruzamento na tabela 'tb_ecoshift_geo_alertas'...{RESETAR}")
        df_db = df[['id', 'siglaTipo', 'numero', 'ano', 'eco_score_neural', 'entidades_locais', 'entidades_orgaos', 'alerta_critico_desmatamento']]
        df_db.to_sql('tb_ecoshift_geo_alertas', self.engine, if_exists='replace', index=False)
        print(f"{VERDE}[OK] Mapeamento concluído com sucesso.{RESETAR}")

        return df

if __name__ == "__main__":
    extrator = GeoEntityExtractor()
    if extrator.engine:
        df_final = extrator.extrair_e_cruzar_entidades()
        
        # Exibir o painel executivo com os alertas ativados
        print(f"\n{CIANO}>>> ALERTAS DE RISCO GEOESPACIAL DETECTADOS (CRUZAMENTO IBAMA) <<<{RESETAR}")
        
        # Filtra apenas os alertas críticos e ordena pelo novo score
        df_alertas = df_final[df_final['alerta_critico_desmatamento'] == True].sort_values(by='eco_score_neural', ascending=False)
        
        if not df_alertas.empty:
            for index, row in df_alertas.head(5).iterrows():
                print(f" {VERMELHO}[ALERTA CRÍTICO]{RESETAR} {row['siglaTipo']} {row['numero']}/{row['ano']} | Novo Score Ajustado: {row['eco_score_neural']:.1f}%")
                print(f" {AMARELO}📍 Área Embargada Detectada:{RESETAR} {row['entidades_locais']}")
                print(f" {AMARELO}🏛️ Órgãos Envolvidos:{RESETAR} {row['entidades_orgaos']}")
                print(f" Resumo: {row['ementa'][:120]}...\n")
        else:
            print(f"{VERDE}Nenhum projeto em tramitação atingiu áreas embargadas do IBAMA nesta janela de análise.{RESETAR}")