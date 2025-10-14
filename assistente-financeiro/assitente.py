# === Mini Assistente para Análise de Ativos ===
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import re

class AssistenteAtivos:
    def __init__(self, ativos=None):
        if ativos is None:
            self.ativos = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'ABEV3.SA']
        else:
            self.ativos = ativos
        
        self.dados = None
        self.carregar_dados()
        self.processar_metricas()
    
    def carregar_dados(self):
        """Carrega e processa os dados dos ativos"""
        print("📥 Carregando dados dos ativos...")
        dados_brutos = yf.download(self.ativos, period='2y')[['Close', 'Volume']]
        
        # Transformar para formato longo
        self.dados = dados_brutos.stack(level=1).reset_index()
        self.dados.rename(columns={
            'Date': 'Data',
            'Ticker': 'Ativo',
            'Close': 'Preço',
            'Volume': 'Volume'
        }, inplace=True)
        
        # Calcular retornos diários
        for ativo in self.ativos:
            mask = self.dados['Ativo'] == ativo
            self.dados.loc[mask, 'Retorno_Diario'] = self.dados.loc[mask, 'Preço'].pct_change()
    
    def processar_metricas(self):
        """Calcula métricas principais para consultas rápidas"""
        self.metricas = {}
        
        for ativo in self.ativos:
            df_ativo = self.dados[self.dados['Ativo'] == ativo].copy()
            
            self.metricas[ativo] = {
                'retorno_total': (df_ativo['Preço'].iloc[-1] / df_ativo['Preço'].iloc[0] - 1) * 100,
                'volatilidade': df_ativo['Retorno_Diario'].std() * 100,
                'preco_atual': df_ativo['Preço'].iloc[-1],
                'preco_min': df_ativo['Preço'].min(),
                'preco_max': df_ativo['Preço'].max(),
                'volume_medio': df_ativo['Volume'].mean()
            }
    
    def consultar_volatilidade(self, periodo=None):
        """Retorna ativos ordenados por volatilidade"""
        volatilidades = {ativo: info['volatilidade'] for ativo, info in self.metricas.items()}
        sorted_vol = sorted(volatilidades.items(), key=lambda x: x[1], reverse=True)
        
        resultado = "📊 RANKING DE VOLATILIDADE:\n"
        for i, (ativo, vol) in enumerate(sorted_vol, 1):
            resultado += f"{i}º {ativo}: {vol:.2f}%\n"
        
        return resultado
    
    def consultar_retorno_total(self):
        """Retorna ativos ordenados por retorno total"""
        retornos = {ativo: info['retorno_total'] for ativo, info in self.metricas.items()}
        sorted_ret = sorted(retornos.items(), key=lambda x: x[1], reverse=True)
        
        resultado = "📈 RANKING DE RETORNO TOTAL:\n"
        for i, (ativo, ret) in enumerate(sorted_ret, 1):
            sinal = "📈" if ret > 0 else "📉"
            resultado += f"{i}º {ativo}: {ret:.2f}% {sinal}\n"
        
        return resultado
    
    def consultar_precos(self, tipo='atual'):
        """Consulta preços atual, mínimo ou máximo"""
        if tipo == 'atual':
            resultado = "💰 PREÇOS ATUAIS:\n"
            for ativo, info in self.metricas.items():
                resultado += f"{ativo}: R$ {info['preco_atual']:.2f}\n"
        
        elif tipo == 'minimo':
            resultado = "📉 PREÇOS MÍNIMOS:\n"
            for ativo, info in self.metricas.items():
                resultado += f"{ativo}: R$ {info['preco_min']:.2f}\n"
        
        elif tipo == 'maximo':
            resultado = "📈 PREÇOS MÁXIMOS:\n"
            for ativo, info in self.metricas.items():
                resultado += f"{ativo}: R$ {info['preco_max']:.2f}\n"
        
        return resultado
    
    def consultar_volume(self):
        """Consulta volumes médios"""
        volumes = {ativo: info['volume_medio'] for ativo, info in self.metricas.items()}
        sorted_vol = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
        
        resultado = "🔄 VOLUMES MÉDIOS DIÁRIOS:\n"
        for i, (ativo, vol) in enumerate(sorted_vol, 1):
            vol_milhoes = vol / 1_000_000
            resultado += f"{i}º {ativo}: {vol_milhoes:.1f}M ações\n"
        
        return resultado
    
    def consultar_melhor_pior(self, metrica='retorno'):
        """Retorna melhor e pior ativo por métrica"""
        if metrica == 'retorno':
            valores = {ativo: info['retorno_total'] for ativo, info in self.metricas.items()}
            melhor = max(valores.items(), key=lambda x: x[1])
            pior = min(valores.items(), key=lambda x: x[1])
            desc = "retorno"
        
        elif metrica == 'volatilidade':
            valores = {ativo: info['volatilidade'] for ativo, info in self.metricas.items()}
            melhor = min(valores.items(), key=lambda x: x[1])  # Menor volatilidade é melhor
            pior = max(valores.items(), key=lambda x: x[1])
            desc = "volatilidade"
        
        resultado = f"🏆 MELHOR E PIOR POR {desc.upper()}:\n"
        resultado += f"🥇 Melhor: {melhor[0]} ({melhor[1]:.2f}%)\n"
        resultado += f"📉 Pior: {pior[0]} ({pior[1]:.2f}%)\n"
        
        return resultado
    
    def consultar_correlacao(self, ativo1=None, ativo2=None):
        """Calcula correlação entre ativos"""
        if ativo1 is None or ativo2 is None:
            # Retorna matriz completa simplificada
            df_pivot = self.dados.pivot(index='Data', columns='Ativo', values='Preço')
            correl_matrix = df_pivot.corr()
            
            resultado = "🔗 MATRIZ DE CORRELAÇÃO:\n"
            for i, ativo_i in enumerate(self.ativos):
                linha = f"{ativo_i}: "
                correlacoes = []
                for j, ativo_j in enumerate(self.ativos):
                    if i != j:
                        correl = correl_matrix.loc[ativo_i, ativo_j]
                        correlacoes.append(f"{ativo_j}({correl:.2f})")
                resultado += linha + " | ".join(correlacoes) + "\n"
        else:
            # Correlação específica entre dois ativos
            df_pivot = self.dados.pivot(index='Data', columns='Ativo', values='Preço')
            correl = df_pivot[ativo1].corr(df_pivot[ativo2])
            
            if correl > 0.7:
                intensidade = "FORTE POSITIVA"
            elif correl > 0.3:
                intensidade = "MODERADA POSITIVA" 
            elif correl > -0.3:
                intensidade = "FRACA"
            elif correl > -0.7:
                intensidade = "MODERADA NEGATIVA"
            else:
                intensidade = "FORTE NEGATIVA"
                
            resultado = f"🔗 CORRELAÇÃO {ativo1} × {ativo2}:\n"
            resultado += f"Valor: {correl:.2f}\n"
            resultado += f"Intensidade: {intensidade}\n"
        
        return resultado
    
    def consultar_resumo_ativo(self, ativo):
        """Resumo completo de um ativo específico"""
        if ativo not in self.metricas:
            return f"❌ Ativo {ativo} não encontrado"
        
        info = self.metricas[ativo]
        resultado = f"📋 RESUMO {ativo}:\n"
        resultado += f"💰 Preço Atual: R$ {info['preco_atual']:.2f}\n"
        resultado += f"📈 Retorno Total: {info['retorno_total']:.2f}%\n"
        resultado += f"📊 Volatilidade: {info['volatilidade']:.2f}%\n"
        resultado += f"⬆️  Preço Máximo: R$ {info['preco_max']:.2f}\n"
        resultado += f"⬇️  Preço Mínimo: R$ {info['preco_min']:.2f}\n"
        resultado += f"🔄 Volume Médio: {info['volume_medio']/1_000_000:.1f}M ações\n"
        
        return resultado
    
    def processar_pergunta(self, pergunta):
        """Processa perguntas em linguagem natural"""
        pergunta = pergunta.lower().strip()
        
        # Padrões de reconhecimento
        if any(palavra in pergunta for palavra in ['volatil', 'risco', 'oscila']):
            return self.consultar_volatilidade()
        
        elif any(palavra in pergunta for palavra in ['retorno', 'desempenho', 'lucro', 'rendimento']):
            if 'melhor' in pergunta or 'pior' in pergunta:
                return self.consultar_melhor_pior('retorno')
            return self.consultar_retorno_total()
        
        elif any(palavra in pergunta for palavra in ['preço', 'valor', 'cotação']):
            if 'mínimo' in pergunta or 'menor' in pergunta:
                return self.consultar_precos('minimo')
            elif 'máximo' in pergunta or 'maior' in pergunta:
                return self.consultar_precos('maximo')
            else:
                return self.consultar_precos('atual')
        
        elif any(palavra in pergunta for palavra in ['volume', 'negocia']):
            return self.consultar_volume()
        
        elif any(palavra in pergunta for palavra in ['correlação', 'relação', 'diversificação']):
            # Tentar extrair ativos específicos da pergunta
            ativos_encontrados = []
            for ativo in self.ativos:
                if ativo.lower() in pergunta:
                    ativos_encontrados.append(ativo)
            
            if len(ativos_encontrados) >= 2:
                return self.consultar_correlacao(ativos_encontrados[0], ativos_encontrados[1])
            else:
                return self.consultar_correlacao()
        
        elif any(palavra in pergunta for palavra in ['resumo', 'info', 'detalhe']):
            for ativo in self.ativos:
                if ativo.lower() in pergunta:
                    return self.consultar_resumo_ativo(ativo)
            return "❌ Especifique qual ativo deseja o resumo (ex: 'resumo PETR4.SA')"
        
        elif 'melhor' in pergunta and 'pior' in pergunta:
            if 'volatilidade' in pergunta:
                return self.consultar_melhor_pior('volatilidade')
            else:
                return self.consultar_melhor_pior('retorno')
        
        else:
            return self.mostrar_ajuda()
    
    def mostrar_ajuda(self):
        """Mostra exemplos de perguntas possíveis"""
        ajuda = """
🤖 ASSISTENTE DE ATIVOS - COMANDOS DISPONÍVEIS:

📊 **Desempenho e Risco:**
• "Qual ativo teve maior retorno?"
• "Qual a volatilidade dos ativos?" 
• "Melhor e pior desempenho"
• "Qual ativo é mais arriscado?"

💰 **Preços:**
• "Quais os preços atuais?"
• "Preços mínimos dos ativos"
• "Preços máximos históricos"

🔗 **Relações:**
• "Correlação entre os ativos"
• "Correlação PETR4 e VALE3"
• "Diversificação da carteira"

📈 **Resumos:**
• "Resumo do PETR4.SA"
• "Informações da VALE3.SA"
• "Detalhes do ITUB4.SA"

🔄 **Volume:**
• "Volume de negociação"
• "Ativos mais negociados"

💡 **Exemplos:**
• "Qual ativo teve maior volatilidade?"
• "Mostre o melhor e pior retorno"
• "Preços atuais dos ativos"
• "Correlação entre PETR4 e VALE3"
• "Resumo do ITUB4.SA"
"""
        return ajuda

# === INTERFACE DO ASSISTENTE ===
def executar_assistente():
    """Loop interativo do assistente"""
    print("🚀 INICIANDO ASSISTENTE DE ATIVOS...")
    assistente = AssistenteAtivos()
    
    print("\n" + "="*60)
    print("🤖 ASSISTENTE FINANCEIRO PRONTO!")
    print("="*60)
    print(assistente.mostrar_ajuda())
    
    while True:
        print("\n" + "-"*40)
        pergunta = input("💬 Faça sua pergunta (ou 'sair' para encerrar): ").strip()
        
        if pergunta.lower() in ['sair', 'exit', 'quit', 'fim']:
            print("👋 Até mais!")
            break
        
        if pergunta == '':
            continue
            
        resposta = assistente.processar_pergunta(pergunta)
        print(f"\n{resposta}")

# === EXECUÇÃO RÁPIDA COM EXEMPLOS ===
def demonstracao_rapida():
    """Mostra exemplos rápidos do assistente"""
    print("🎯 DEMONSTRAÇÃO RÁPIDA DO ASSISTENTE")
    assistente = AssistenteAtivos()
    
    exemplos = [
        "Qual ativo teve maior volatilidade?",
        "Melhor e pior desempenho",
        "Preços atuais dos ativos", 
        "Correlação entre PETR4 e VALE3",
        "Resumo do ITUB4.SA",
        "Volume de negociação"
    ]
    
    for exemplo in exemplos:
        print(f"\n💬 Pergunta: {exemplo}")
        print(f"🤖 Resposta: {assistente.processar_pergunta(exemplo)}")
        print("-" * 50)

# === MENU PRINCIPAL ===
if __name__ == "__main__":
    print("="*60)
    print("           MINI ASSISTENTE DE ATIVOS B3")
    print("="*60)
    
    while True:
        print("\n📋 OPÇÕES:")
        print("1. 🗣️  Modo Interativo (Conversar com o assistente)")
        print("2. 🎯 Demonstração Rápida (Ver exemplos)")
        print("3. ❌ Sair")
        
        opcao = input("\nEscolha uma opção (1-3): ").strip()
        
        if opcao == "1":
            executar_assistente()
        elif opcao == "2":
            demonstracao_rapida()
        elif opcao == "3":
            print("👋 Encerrando programa...")
            break
        else:
            print("❌ Opção inválida. Tente novamente.")