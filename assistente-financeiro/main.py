# === CONFIGURAÇÕES E IMPORTAÇÕES ===
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import json
from datetime import datetime

# Configuração para melhor visualização
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# === COLETA DE DADOS FINANCEIROS ===
def coletar_dados_ativos():
    """Coleta dados dos ativos via Yahoo Finance"""
    print("📥 Baixando dados dos ativos...")
    
    # Lista de ativos
    ativos = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'ABEV3.SA']
    
    # Baixar todos os ativos de uma vez
    dados = yf.download(ativos, period='2y')[['Close', 'Volume']]
    
    # Empilhar os níveis de coluna para transformar em formato longo
    dados = dados.stack(level=1).reset_index()
    
    # Renomear colunas
    dados.rename(columns={
        'Date': 'Data',
        'Ticker': 'Ativo',
        'Close': 'Preço de Fechamento',
        'Volume': 'Volume'
    }, inplace=True)
    
    # Salvar dados em CSV
    dados.to_csv('dados_ativos.csv', index=False)
    print("✅ Dados salvos em 'dados_ativos.csv'")
    
    return dados, ativos

# === CÁLCULO DE RENTABILIDADE E VOLATILIDADE ===
def calcular_rentabilidade_volatilidade(dados, ativos):
    """Calcula rentabilidade e volatilidade detalhada dos ativos"""
    
    resultados_detalhados = []
    
    for ativo in ativos:
        df = dados[dados['Ativo'] == ativo].copy()
        
        # Rentabilidade Diária
        df['Retorno Diário'] = df['Preço de Fechamento'].pct_change()
        df['Retorno Diário (%)'] = df['Retorno Diário'] * 100
        
        # Rentabilidade Acumulada (conforme fórmula solicitada)
        preco_inicial = df['Preço de Fechamento'].iloc[0]
        preco_final = df['Preço de Fechamento'].iloc[-1]
        rentab_acumulada = (preco_final / preco_inicial - 1) * 100
        
        # Cálculo DETALHADO da Volatilidade (passo a passo)
        retornos = df['Retorno Diário'].dropna()
        
        # Passo 1: Média dos retornos
        media_retornos = retornos.mean()
        
        # Passo 2: Subtrair a média de cada retorno
        desvios = retornos - media_retornos
        
        # Passo 3: Elevar ao quadrado as diferenças
        desvios_quad = desvios ** 2
        
        # Passo 4: Média das diferenças quadradas (Variância)
        variancia = desvios_quad.mean()
        
        # Passo 5: Raiz quadrada da variância (Desvio Padrão/Volatilidade)
        volatilidade = np.sqrt(variancia) * 100
        
        # Estatísticas adicionais
        retorno_medio_diario = media_retornos * 100
        melhor_dia = df['Retorno Diário (%)'].max()
        pior_dia = df['Retorno Diário (%)'].min()
        
        resultados_detalhados.append([
            ativo, 
            rentab_acumulada, 
            volatilidade,
            retorno_medio_diario,
            melhor_dia,
            pior_dia,
            preco_inicial,
            preco_final
        ])
        
        print(f"\n🔍 ANÁLISE DETALHADA - {ativo}:")
        print(f"   Preço Inicial: R$ {preco_inicial:.2f}")
        print(f"   Preço Final: R$ {preco_final:.2f}")
        print(f"   Rentabilidade Acumulada: {rentab_acumulada:.2f}%")
        print(f"   Volatilidade Diária: {volatilidade:.2f}%")
        print(f"   Retorno Médio Diário: {retorno_medio_diario:.4f}%")
        print(f"   Melhor Dia: {melhor_dia:.2f}%")
        print(f"   Pior Dia: {pior_dia:.2f}%")
    
    # DataFrame com resultados
    colunas = ['Ativo', 'Rentabilidade Acumulada (%)', 'Volatilidade Diária (%)', 
               'Retorno Médio Diário (%)', 'Melhor Dia (%)', 'Pior Dia (%)',
               'Preço Inicial (R$)', 'Preço Final (R$)']
    
    resultados_df = pd.DataFrame(resultados_detalhados, columns=colunas)
    
    return resultados_df

#  === ANÁLISE DE CORRELAÇÃO ENTRE ATIVOS ===
def analisar_correlacao(dados, ativo1='PETR4.SA', ativo2='VALE3.SA'):
    """Analisa e plota correlação entre dois ativos"""
    
    # Preparar dados para correlação
    df_corr = dados.pivot(index='Data', columns='Ativo', values='Preço de Fechamento')
    
    # Calcular correlação (equivalente à função CORREL do Excel)
    correlacao = df_corr[ativo1].corr(df_corr[ativo2])
    
    # Interpretação da correlação
    if correlacao > 0.7:
        interpretacao = "FORTE CORRELAÇÃO POSITIVA"
    elif correlacao > 0.3:
        interpretacao = "CORRELAÇÃO POSITIVA MODERADA"
    elif correlacao > -0.3:
        interpretacao = "CORRELAÇÃO FRACA"
    elif correlacao > -0.7:
        interpretacao = "CORRELAÇÃO NEGATIVA MODERADA"
    else:
        interpretacao = "FORTE CORRELAÇÃO NEGATIVA"
    
    print(f"Correlação entre {ativo1} e {ativo2}: {correlacao:.4f}")
    print(f"Interpretação: {interpretacao}")
    
    # Gráfico de Correlação entre dois ativos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Dispersão entre os dois ativos
    ax1.scatter(df_corr[ativo1], df_corr[ativo2], alpha=0.6, color='blue')
    ax1.set_xlabel(f'Preço {ativo1}')
    ax1.set_ylabel(f'Preço {ativo2}')
    ax1.set_title(f'Correlação: {ativo1} vs {ativo2}\n(r = {correlacao:.3f})')
    ax1.grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(df_corr[ativo1], df_corr[ativo2], 1)
    p = np.poly1d(z)
    ax1.plot(df_corr[ativo1], p(df_corr[ativo1]), "r--", alpha=0.8)
    
    # Gráfico 2: Evolução conjunta dos preços
    ax2.plot(df_corr.index, df_corr[ativo1], label=ativo1, linewidth=2)
    ax2.plot(df_corr.index, df_corr[ativo2], label=ativo2, linewidth=2)
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Preço (R$)')
    ax2.set_title('Evolução dos Preços - Comparação')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('correlacao_ativos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlacao, interpretacao

#  === GRÁFICO RETORNO X VOLATILIDADE ===
def plotar_retorno_vs_volatilidade(resultados_df):
    """Cria gráfico comparando retorno e volatilidade dos ativos"""
    
    plt.figure(figsize=(12, 8))
    
    # Criar scatter plot com mais informações
    scatter = plt.scatter(resultados_df['Volatilidade Diária (%)'], 
                         resultados_df['Rentabilidade Acumulada (%)'],
                         s=200, alpha=0.7, 
                         c=resultados_df['Rentabilidade Acumulada (%)'],
                         cmap='RdYlGn')
    
    # Adicionar anotações para cada ativo
    for i, row in resultados_df.iterrows():
        plt.annotate(row['Ativo'], 
                    (row['Volatilidade Diária (%)'], row['Rentabilidade Acumulada (%)']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold')
        
        # Adicionar retorno acumulado como texto
        plt.annotate(f"{row['Rentabilidade Acumulada (%)']:.1f}%",
                    (row['Volatilidade Diária (%)'], row['Rentabilidade Acumulada (%)']),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=10, color='gray')
    
    plt.xlabel('Volatilidade Diária (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Rentabilidade Acumulada (%)', fontsize=12, fontweight='bold')
    plt.title('Relação Retorno x Volatilidade - Análise de Ativos\n(Últimos 2 Anos)', 
              fontsize=14, fontweight='bold')
    
    # Adicionar linhas de referência
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Retorno Zero')
    plt.axvline(x=resultados_df['Volatilidade Diária (%)'].mean(), 
               color='blue', linestyle='--', alpha=0.5, label='Volatilidade Média')
    
    plt.colorbar(scatter, label='Rentabilidade Acumulada (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar quadrantes para análise
    x_median = resultados_df['Volatilidade Diária (%)'].median()
    y_median = resultados_df['Rentabilidade Acumulada (%)'].median()
    
    plt.text(x_median*0.8, y_median*1.8, 'Alto Retorno\nAlto Risco', 
             fontsize=10, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.3))
    
    plt.text(x_median*0.8, y_median*0.5, 'Baixo Retorno\nAlto Risco', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.3))
    
    plt.text(x_median*1.2, y_median*1.8, 'Alto Retorno\nBaixo Risco', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
    
    plt.text(x_median*1.2, y_median*0.5, 'Baixo Retorno\nBaixo Risco', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('retorno_vs_volatilidade_detalhado.png', dpi=300, bbox_inches='tight')
    plt.show()

# === RELATÓRIO FINAL E RESUMO ===
def gerar_relatorio_final(resultados_df, correlacao, interpretacao, ativo1, ativo2):
    """Gera relatório final com resumo da análise"""
    
    print("\n" + "="*80)
    print("🎯 RESUMO FINAL DA ANÁLISE")
    print("="*80)
    print(f"📈 Melhor Rentabilidade: {resultados_df.loc[resultados_df['Rentabilidade Acumulada (%)'].idxmax()]['Ativo']}")
    print(f"📉 Pior Rentabilidade: {resultados_df.loc[resultados_df['Rentabilidade Acumulada (%)'].idxmin()]['Ativo']}")
    print(f"⚡ Maior Volatilidade: {resultados_df.loc[resultados_df['Volatilidade Diária (%)'].idxmax()]['Ativo']}")
    print(f"🛡️  Menor Volatilidade: {resultados_df.loc[resultados_df['Volatilidade Diária (%)'].idxmin()]['Ativo']}")
    print(f"🔗 Correlação {ativo1}/{ativo2}: {correlacao:.3f} ({interpretacao})")
    
    print("\n✅ Análise concluída! Gráficos salvos como:")
    print("   - correlacao_ativos.png")
    print("   - retorno_vs_volatilidade_detalhado.png")

# === FUNÇÃO PRINCIPAL - EXECUÇÃO DO PROGRAMA ===
def main():
    """Função principal que orquestra toda a análise"""
    
    # SEÇÃO 1: Coleta de Dados
    print("="*80)
    print("📊 ANÁLISE COMPLETA DOS ATIVOS")
    print("="*80)
    
    dados, ativos = coletar_dados_ativos()
    
    # SEÇÃO 2: Cálculo de Rentabilidade e Volatilidade
    print("\n" + "="*80)
    print("📈 CÁLCULO DE RENTABILIDADE E VOLATILIDADE")
    print("="*80)
    
    resultados_df = calcular_rentabilidade_volatilidade(dados, ativos)
    
    # Exibir resultados gerais
    print("\n" + "="*80)
    print("📊 RESUMO GERAL DOS RESULTADOS")
    print("="*80)
    print(resultados_df.round(2))
    
    # Identificar ativo com maior volatilidade
    ativo_maior_vol = resultados_df.loc[resultados_df['Volatilidade Diária (%)'].idxmax()]
    print(f"\n🔥 ATIVO COM MAIOR VOLATILIDADE:")
    print(f"   {ativo_maior_vol['Ativo']}: {ativo_maior_vol['Volatilidade Diária (%)']:.2f}%")
    
    # SEÇÃO 3: Análise de Correlação
    print("\n" + "="*80)
    print("🔗 ANÁLISE DE CORRELAÇÃO")
    print("="*80)
    
    ativo1, ativo2 = 'PETR4.SA', 'VALE3.SA'
    correlacao, interpretacao = analisar_correlacao(dados, ativo1, ativo2)
    
    # SEÇÃO 4: Gráfico Retorno vs Volatilidade
    print("\n" + "="*80)
    print("📊 GRÁFICO: RETORNO X VOLATILIDADE")
    print("="*80)
    
    plotar_retorno_vs_volatilidade(resultados_df)
    
    # SEÇÃO 5: Relatório Final
    gerar_relatorio_final(resultados_df, correlacao, interpretacao, ativo1, ativo2)

# === EXECUÇÃO DO PROGRAMA ===
if __name__ == "__main__":
    main()