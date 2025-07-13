# app_doers
Interpreta DOE
# 📄 Consulta Inteligente de PDFs

Sistema avançado para busca semântica em documentos PDF extensos usando NLP, embeddings e RAG.

## 🎯 Funcionalidades

### ✅ **Extração Robusta de Texto**
- **PyMuPDF (primário)**: Extração mais precisa com informações de posição
- **PyPDF2 (fallback)**: Backup para PDFs problemáticos
- **Limpeza automática**: Remove caracteres especiais e normaliza texto

### ✅ **Busca Híbrida Inteligente**
- **Embeddings semânticos**: Encontra conteúdo similar mesmo com palavras diferentes
- **TF-IDF**: Busca por palavras-chave tradicionais
- **Busca combinada**: Melhor precisão com abordagem híbrida

### ✅ **Chunking Inteligente**
- **Chunks com overlap**: Preserva contexto entre seções
- **Divisão por sentenças**: Mantém coerência semântica
- **Tamanho configurável**: Adaptável ao tipo de documento

### ✅ **Sistema RAG (Retrieval-Augmented Generation)**
- **Respostas automáticas**: Usando OpenAI GPT
- **Contexto preservado**: Referências às páginas originais
- **Opcional**: Funciona sem API key

## 🚀 Instalação e Uso

### 1. **Preparação do Ambiente**
```bash
# Salve todos os arquivos em: C:\Users\lucas\Downloads\
# app.py, requirements.txt, setup.bat, README.md

cd C:\Users\lucas\Downloads\
```

### 2. **Instalação Automática**
```bash
# Execute o script de instalação
setup.bat
```

### 3. **Instalação Manual** (se preferir)
```bash
# Crie ambiente virtual
python -m venv venv

# Ative o ambiente (Windows)
venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt
```

### 4. **Execução**
```bash
# Ative o ambiente virtual
venv\Scripts\activate

# Execute o aplicativo
streamlit run app.py
```

## 📋 Dependências

- **streamlit**: Interface web
- **PyMuPDF**: Extração de PDF principal
- **PyPDF2**: Extração de PDF backup
- **sentence-transformers**: Embeddings multilíngues
- **scikit-learn**: TF-IDF e similaridade
- **numpy**: Computação numérica
- **pandas**: Manipulação de dados
- **openai**: API OpenAI (opcional)

## 🔧 Configurações Avançadas

### **Chunking**
```python
chunk_size = 1000      # Tamanho do chunk (caracteres)
chunk_overlap = 200    # Overlap entre chunks
```

### **Busca**
```python
# Pesos da busca híbrida
embedding_weight = 0.6  # Busca semântica
tfidf_weight = 0.4     # Busca por palavras-chave
```

### **OpenAI RAG**
- Configure sua API key na sidebar
- Modelo padrão: `gpt-3.5-turbo`
- Temperatura: 0.3 (mais preciso)

## 📊 Como Usar

### 1. **Upload do PDF**
- Clique em "Choose a file" e selecione seu PDF
- Aguarde o processamento automático

### 2. **Configuração**
- Ajuste o tamanho dos chunks na sidebar
- Configure API key OpenAI se desejado

### 3. **Busca**
- Digite sua consulta no campo de busca
- Ajuste número de resultados
- Veja resultados com destaque de termos

### 4. **RAG (Opcional)**
- Clique em "Gerar Resposta Automática"
- Obtenha resumo inteligente dos resultados

## 🎯 Exemplos de Uso

### **Diários Oficiais**
```
"contratos de prestação de serviços em 2024"
"licitações para obras públicas"
"nomeações de servidores públicos"
```

### **Contratos Jurídicos**
```
"cláusulas de rescisão contratual"
"penalidades por descumprimento"
"condições de pagamento"
```

### **Documentos Técnicos**
```
"especificações técnicas de equipamentos"
"procedimentos de segurança"
"normas de qualidade"
```

## 🛠️ Arquitetura do Sistema

```
PDF → Extração → Chunking → Indexação → Busca → Resultados
                                      ↓
                                    RAG → Resposta
```

### **Pipeline de Processamento**
1. **Extração**: PyMuPDF + PyPDF2
2. **Limpeza**: Normalização e remoção de ruído
3. **Chunking**: Divisão inteligente com overlap
4. **Embeddings**: Sentence Transformers multilíngue
5. **Indexação**: TF-IDF + Embeddings
6. **Busca**: Híbrida com scores combinados
7. **RAG**: Geração de respostas contextuais

## 🔍 Otimizações para PDFs Grandes

### **Memória**
- Processamento incremental de páginas
- Garbage collection automático
- Chunks otimizados

### **Performance**
- Índices pré-computados
- Cache de embeddings
- Busca vetorizada

### **Qualidade**
- Múltiplos extratores de PDF
- Limpeza robusta de texto
- Chunking preservando contexto

## 🚨 Solução de Problemas

### **Erro de Extração**
```
Problema: "Não foi possível extrair texto do PDF"
Solução: PDF pode estar corrompido ou ter proteção
```

### **Busca Lenta**
```
Problema: Demora para processar consultas
Solução: Reduza tamanho dos chunks ou número de resultados
```

### **Resultados Irrelevantes**
