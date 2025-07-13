import streamlit as st
import os
import tempfile
import pickle
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional
import logging

# Importações para processamento de PDF e NLP
import PyPDF2
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Para funcionalidades avançadas com LLM (opcional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Classe para processar e extrair texto de PDFs"""
    
    def __init__(self):
        self.text_content = []
        self.metadata = {}
    
    def extract_text_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Extrai texto usando PyMuPDF (mais robusto para PDFs complexos)"""
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extrai texto com informações de posição
                text_dict = page.get_text("dict")
                page_text = ""
                
                # Processa blocos de texto
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"] + " "
                            page_text += line_text.strip() + "\n"
                
                # Limpa e normaliza o texto
                page_text = self.clean_text(page_text)
                
                pages_content.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
            
            doc.close()
            return pages_content
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto com PyMuPDF: {e}")
            return []
    
    def extract_text_pypdf2(self, pdf_path: str) -> List[Dict]:
        """Extrai texto usando PyPDF2 (fallback)"""
        try:
            pages_content = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Limpa e normaliza o texto
                    text = self.clean_text(text)
                    
                    pages_content.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'char_count': len(text)
                    })
            
            return pages_content
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto com PyPDF2: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto extraído"""
        # Remove caracteres especiais e normaliza espaços
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\%\$\#\&\*\+\=\<\>\|\\]', ' ', text)
        text = text.strip()
        
        # Remove linhas muito curtas (provavelmente lixo)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        
        return '\n'.join(cleaned_lines)
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Processa o PDF e extrai texto"""
        st.info("Extraindo texto do PDF...")
        
        # Tenta primeiro com PyMuPDF
        pages_content = self.extract_text_pymupdf(pdf_path)
        
        # Se falhar, usa PyPDF2
        if not pages_content:
            st.warning("PyMuPDF falhou, tentando com PyPDF2...")
            pages_content = self.extract_text_pypdf2(pdf_path)
        
        if not pages_content:
            raise Exception("Não foi possível extrair texto do PDF")
        
        # Salva metadados
        self.metadata = {
            'total_pages': len(pages_content),
            'total_characters': sum(page['char_count'] for page in pages_content),
            'processed_at': datetime.now().isoformat()
        }
        
        self.text_content = pages_content
        return pages_content

class TextChunker:
    """Classe para dividir texto em chunks otimizados"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_chunks(self, pages_content: List[Dict]) -> List[Dict]:
        """Cria chunks do texto com overlap"""
        chunks = []
        chunk_id = 0
        
        for page_data in pages_content:
            page_num = page_data['page_number']
            text = page_data['text']
            
            # Divide o texto em sentenças para melhor chunking
            sentences = self.split_into_sentences(text)
            
            current_chunk = ""
            current_sentences = []
            
            for sentence in sentences:
                # Verifica se adicionar a sentença excede o tamanho do chunk
                if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                    # Cria chunk atual
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk.strip(),
                        'page_number': page_num,
                        'sentences': current_sentences.copy(),
                        'char_count': len(current_chunk)
                    })
                    
                    chunk_id += 1
                    
                    # Inicia novo chunk com overlap
                    overlap_text = self.get_overlap_text(current_sentences)
                    current_chunk = overlap_text + sentence
                    current_sentences = self.get_overlap_sentences(current_sentences) + [sentence]
                else:
                    current_chunk += sentence
                    current_sentences.append(sentence)
            
            # Adiciona último chunk da página
            if current_chunk.strip():
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'page_number': page_num,
                    'sentences': current_sentences,
                    'char_count': len(current_chunk)
                })
                chunk_id += 1
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Divide texto em sentenças"""
        # Regex para dividir sentenças (pode ser aprimorado)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() + ' ' for s in sentences if s.strip()]
    
    def get_overlap_text(self, sentences: List[str]) -> str:
        """Obtém texto de overlap baseado no tamanho"""
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= self.chunk_overlap:
                overlap_text = sentence + overlap_text
            else:
                break
        return overlap_text
    
    def get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Obtém sentenças de overlap"""
        overlap_sentences = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences

class IntelligentSearch:
    """Classe para busca inteligente usando embeddings e TF-IDF"""
    
    def __init__(self):
        self.embeddings_model = None
        self.tfidf_vectorizer = None
        self.chunks = []
        self.chunk_embeddings = None
        self.tfidf_matrix = None
        
    def initialize_models(self):
        """Inicializa os modelos de busca"""
        if self.embeddings_model is None:
            with st.spinner("Carregando modelo de embeddings..."):
                # Usa um modelo multilíngue para português
                self.embeddings_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=None,  # Mantém stop words para português
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8
            )
    
    def index_chunks(self, chunks: List[Dict]):
        """Indexa chunks para busca"""
        self.chunks = chunks
        self.initialize_models()
        
        if not chunks:
            return
        
        # Extrai textos dos chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Cria embeddings
        with st.spinner("Criando embeddings..."):
            self.chunk_embeddings = self.embeddings_model.encode(texts)
        
        # Cria matriz TF-IDF
        with st.spinner("Criando índice TF-IDF..."):
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Realiza busca híbrida (embeddings + TF-IDF)"""
        if not self.chunks:
            return []
        
        # Busca por embeddings (semântica)
        query_embedding = self.embeddings_model.encode([query])
        embedding_similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Busca por TF-IDF (palavras-chave)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Combina as similaridades (híbrido)
        combined_similarities = 0.6 * embedding_similarities + 0.4 * tfidf_similarities
        
        # Obtém top resultados
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_similarities[idx] > 0.1:  # Threshold mínimo
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(combined_similarities[idx])
                result['embedding_score'] = float(embedding_similarities[idx])
                result['tfidf_score'] = float(tfidf_similarities[idx])
                results.append(result)
        
        return results
    
    def highlight_terms(self, text: str, query: str) -> str:
        """Destaca termos da busca no texto"""
        query_terms = query.lower().split()
        highlighted_text = text
        
        for term in query_terms:
            if len(term) > 2:  # Evita palavras muito curtas
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted_text = pattern.sub(f'**{term}**', highlighted_text)
        
        return highlighted_text

class RAGSystem:
    """Sistema de RAG para respostas automáticas"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        
        if api_key and OPENAI_AVAILABLE:
            openai.api_key = api_key
            self.client = openai
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Gera resposta usando RAG"""
        if not self.client:
            return "Sistema RAG não disponível. Configure uma API key do OpenAI."
        
        # Combina contexto
        context = "\n\n".join([
            f"Página {chunk['page_number']}: {chunk['text'][:500]}..."
            for chunk in context_chunks[:5]  # Limita contexto
        ])
        
        prompt = f"""
        Baseado no seguinte contexto extraído de um documento:

        {context}

        Responda à pergunta: {query}

        Instruções:
        - Use apenas informações do contexto fornecido
        - Seja preciso e objetivo
        - Indique as páginas de referência quando possível
        - Se não houver informação suficiente, diga claramente
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

def main():
    """Função principal da aplicação Streamlit"""
    st.set_page_config(
        page_title="Consulta Inteligente de PDFs",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Consulta Inteligente de PDFs")
    st.markdown("*Sistema avançado para busca em documentos extensos*")
    
    # Inicializa session state
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    
    if 'text_chunker' not in st.session_state:
        st.session_state.text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = IntelligentSearch()
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Configurações de chunking
        chunk_size = st.slider("Tamanho do Chunk", 500, 2000, 1000)
        chunk_overlap = st.slider("Overlap entre Chunks", 50, 500, 200)
        
        # Configuração OpenAI (opcional)
        st.subheader("🤖 RAG com OpenAI")
        openai_key = st.text_input("API Key OpenAI", type="password")
        if openai_key:
            st.session_state.rag_system = RAGSystem(openai_key)
        
        # Estatísticas do documento
        if st.session_state.pdf_processed:
            st.subheader("📊 Estatísticas")
            metadata = st.session_state.pdf_processor.metadata
            st.write(f"**Páginas:** {metadata.get('total_pages', 0)}")
            st.write(f"**Caracteres:** {metadata.get('total_characters', 0):,}")
            st.write(f"**Chunks:** {len(st.session_state.chunks)}")
    
    # Upload de PDF
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")
    
    if uploaded_file is not None:
        # Salva arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Processa PDF
            if st.button("🔄 Processar PDF") or not st.session_state.pdf_processed:
                with st.spinner("Processando PDF..."):
                    # Extrai texto
                    pages_content = st.session_state.pdf_processor.process_pdf(tmp_file_path)
                    
                    # Cria chunks
                    st.session_state.text_chunker.chunk_size = chunk_size
                    st.session_state.text_chunker.chunk_overlap = chunk_overlap
                    chunks = st.session_state.text_chunker.create_chunks(pages_content)
                    st.session_state.chunks = chunks
                    
                    # Indexa para busca
                    st.session_state.search_engine.index_chunks(chunks)
                    
                    st.session_state.pdf_processed = True
                    st.success(f"PDF processado com sucesso! {len(chunks)} chunks criados.")
        
        finally:
            # Remove arquivo temporário
            os.unlink(tmp_file_path)
    
    # Interface de busca
    if st.session_state.pdf_processed:
        st.header("🔍 Busca Inteligente")
        
        # Campo de busca
        query = st.text_input("Digite sua consulta:", placeholder="Ex: contratos de prestação de serviços")
        
        # Configurações de busca
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Número de resultados", 1, 20, 5)
        with col2:
            show_scores = st.checkbox("Mostrar scores de similaridade")
        
        if query:
            # Realiza busca
            results = st.session_state.search_engine.search(query, top_k=num_results)
            
            if results:
                st.subheader(f"📋 Resultados ({len(results)} encontrados)")
                
                # Opção RAG
                if st.button("🤖 Gerar Resposta Automática (RAG)"):
                    with st.spinner("Gerando resposta..."):
                        answer = st.session_state.rag_system.generate_answer(query, results)
                        st.info(f"**Resposta Automática:** {answer}")
                
                # Exibe resultados
                for i, result in enumerate(results):
                    with st.expander(f"📄 Resultado {i+1} - Página {result['page_number']} {f'(Score: {result['similarity_score']:.3f})' if show_scores else ''}"):
                        
                        # Texto destacado
                        highlighted_text = st.session_state.search_engine.highlight_terms(
                            result['text'], query
                        )
                        st.markdown(highlighted_text)
                        
                        # Informações adicionais
                        if show_scores:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Score Geral", f"{result['similarity_score']:.3f}")
                            with col2:
                                st.metric("Score Semântico", f"{result['embedding_score']:.3f}")
                            with col3:
                                st.metric("Score TF-IDF", f"{result['tfidf_score']:.3f}")
                        
                        st.caption(f"Chunk ID: {result['chunk_id']} | Página: {result['page_number']} | Caracteres: {result['char_count']}")
            else:
                st.warning("Nenhum resultado encontrado. Tente termos diferentes.")
    
    # Informações sobre o sistema
    with st.expander("ℹ️ Sobre o Sistema"):
        st.markdown("""
        **Funcionalidades:**
        - ✅ Extração robusta de texto (PyMuPDF + PyPDF2)
        - ✅ Busca híbrida (embeddings + TF-IDF)
        - ✅ Chunking inteligente com overlap
        - ✅ Suporte a documentos extensos
        - ✅ Busca semântica (não apenas palavras exatas)
        - ✅ Sistema RAG para respostas automáticas
        - ✅ Interface amigável
        
        **Tecnologias:**
        - **Sentence Transformers:** Embeddings multilíngues
        - **scikit-learn:** TF-IDF e cosine similarity
        - **PyMuPDF:** Extração robusta de PDF
        - **Streamlit:** Interface web
        - **OpenAI:** RAG (opcional)
        """)

if __name__ == "__main__":
    main()