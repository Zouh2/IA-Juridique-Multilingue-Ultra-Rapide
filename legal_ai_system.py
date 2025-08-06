import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime
from textwrap import dedent
import logging
from pathlib import Path
import pdfplumber
from docx import Document
from functools import wraps
import time
import re
import json
from openai import OpenAI, RateLimitError, APIError
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import concurrent.futures
from threading import Lock
import langdetect
from googletrans import Translator

from agno.models.openai import OpenAIChat
from agno.agent import Agent
from agno.tools import tool
from agno.tools.file import FileTools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration OpenAI API - MULTILANGUAGE OPTIMIZED
OPENAI_API_KEY = "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"
MODEL_NAME = 'gpt-4o'  # Optimisé pour rapidité et multilinguisme

# Configuration optimisée pour la vitesse multilingue
RATE_LIMIT_DELAY = 0.3  # Encore plus rapide
MAX_RETRIES = 2
BACKOFF_FACTOR = 1.5
MAX_CONCURRENT_REQUESTS = 4  # Augmentation pour traitement parallèle

# Configure Tesseract with multilingual support
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Multilingual settings
SUPPORTED_LANGUAGES = {
    'fr': {'name': 'Français', 'tesseract': 'fra', 'direction': 'ltr'},
    'en': {'name': 'English', 'tesseract': 'eng', 'direction': 'ltr'},
    'ar': {'name': 'العربية', 'tesseract': 'ara', 'direction': 'rtl'}
}

# Initialize translator
translator = Translator()

class MultilingualRateLimitHandler:
    def __init__(self, requests_per_minute: int = 120):  # Augmenté pour multilangue
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.lock = Lock()
    
    async def wait_if_needed(self):
        with self.lock:
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]
            if len(self.request_times) >= self.requests_per_minute:
                wait_time = max(0.1, 60 - (now - self.request_times[0]) + 0.1)
                await asyncio.sleep(wait_time)
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60]
            self.request_times.append(now)

rate_limiter = MultilingualRateLimitHandler()

# Cache pour les modèles multilingues
_multilingual_model_cache = {}

def get_multilingual_cached_model():
    if 'multilingual_openai_model' not in _multilingual_model_cache:
        _multilingual_model_cache['multilingual_openai_model'] = OpenAIChat(
            id=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.2,  # Plus bas pour plus de précision multilingue
            max_retries=MAX_RETRIES,
            timeout=45.0
        )
    return _multilingual_model_cache['multilingual_openai_model']

# Détection de langue améliorée
def detect_document_language(text: str) -> str:
    """Détecte la langue principale du document"""
    try:
        # Nettoyer le texte pour la détection
        clean_text = re.sub(r'[^\w\s]', ' ', text[:2000])  # Prendre les premiers 2000 chars
        detected = langdetect.detect(clean_text)
        
        # Mapper les codes de langue
        language_mapping = {
            'fr': 'fr', 'en': 'en', 'ar': 'ar',
            'es': 'en',  # Fallback vers anglais pour espagnol
            'de': 'en',  # Fallback vers anglais pour allemand
        }
        
        return language_mapping.get(detected, 'en')  # Default to English
    except Exception as e:
        logger.warning(f"Erreur détection langue: {e}")
        return 'en'  # Default fallback

# Retry decorator optimisé pour multilangue
def multilingual_retry_with_rate_limit(max_attempts=2, base_delay=0.3):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = 0
            last_error = None
            while attempts < max_attempts:
                try:
                    await rate_limiter.wait_if_needed()
                    return await func(*args, **kwargs)
                except (RateLimitError, APIError) as e:
                    attempts += 1
                    last_error = e
                    if attempts < max_attempts:
                        wait_time = base_delay * (BACKOFF_FACTOR ** (attempts - 1))
                        await asyncio.sleep(wait_time)
                except Exception as e:
                    attempts += 1
                    last_error = e
                    if attempts < max_attempts:
                        await asyncio.sleep(base_delay)
            raise Exception(f"Failed after {max_attempts} attempts: {str(last_error)}")
        return wrapper
    return decorator

# Data structures multilingues
@dataclass
class MultilingualContractAnalysis:
    content: str
    source: str
    timestamp: datetime
    language: str
    language_confidence: float

@dataclass 
class MultilingualLegalIssue:
    clause: str
    section: str
    issue: str
    severity: str
    confidence: float
    language: str
    category: str  # Nouvelle catégorie

@dataclass
class MultilingualNegotiationPoint:
    quoted_clause: str
    why_negotiable: str
    counter_suggestion: str
    leverage_score: float
    language: str
    priority: str  # Nouvelle priorité

@dataclass
class ComplianceAnalysis:
    regulation: str
    compliance_status: str
    gaps: List[str]
    recommendations: List[str]
    language: str

@dataclass
class RiskAssessment:
    financial_risk: float
    legal_risk: float
    operational_risk: float
    overall_risk: float
    mitigation_strategies: List[str]
    language: str

# Fonctions d'extraction optimisées et multilingues
def sanitize_multilingual_text(text: str, language: str = 'en') -> str:
    if not text:
        return ""
    
    # Nettoyage spécifique selon la langue
    if language == 'ar':
        # Préservation des caractères arabes
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\w]', ' ', text)
    else:
        # Nettoyage standard pour français/anglais
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
    
    return text[:12000]  # Augmenté pour multilangue

def extract_pdf_text_multilingual(file_path: str) -> Tuple[str, str]:
    """Extraction PDF multilingue avec détection automatique"""
    try:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            max_pages = min(len(pdf.pages), 25)  # Augmenté à 25 pages
            for page in pdf.pages[:max_pages]:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        if text_parts:
            full_text = ' '.join(text_parts)
            language = detect_document_language(full_text)
            return sanitize_multilingual_text(full_text, language), language
        else:
            # OCR multilingue
            logger.warning("Fallback to multilingual OCR")
            images = convert_from_path(file_path, last_page=2)  # Première 2 pages
            if images:
                # Essayer avec différentes langues OCR
                for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
                    try:
                        text = pytesseract.image_to_string(images[0], lang=lang_info['tesseract'])
                        if len(text.strip()) > 100:  # Si on trouve du texte significatif
                            detected_lang = detect_document_language(text)
                            return sanitize_multilingual_text(text, detected_lang), detected_lang
                    except:
                        continue
            
        return "Impossible d'extraire le texte du PDF", 'en'
    except Exception as e:
        logger.error(f"Erreur extraction PDF multilingue: {str(e)}")
        return f"Erreur d'extraction: {str(e)}", 'en'

def create_multilingual_knowledge_base(file_path: str) -> Tuple[None, str, str]:
    """Base de connaissances multilingue"""
    file_path = Path(file_path)
    try:
        if file_path.suffix.lower() == '.pdf':
            text_content, language = extract_pdf_text_multilingual(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs[:250] if p.text.strip()]
            text_content = ' '.join(paragraphs)
            language = detect_document_language(text_content)
            text_content = sanitize_multilingual_text(text_content, language)
        elif file_path.suffix.lower() in ['.txt', '.md', '.rst']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(60000)
            language = detect_document_language(content)
            text_content = sanitize_multilingual_text(content, language)
        else:
            raise ValueError(f"Format non supporté: {file_path.suffix}")
        
        logger.info(f"Texte extrait: {len(text_content)} chars, Langue: {language}")
        return None, text_content, language
    except Exception as e:
        logger.error(f"Erreur traitement fichier multilingue: {str(e)}")
        raise

# Tools multilingues et avancés
@tool
def advanced_multilingual_section_analyzer(content: str, language: str = 'en') -> Dict[str, Any]:
    """Analyseur de sections multilingue avancé"""
    content = sanitize_multilingual_text(content, language)[:6000]
    
    # Mots-clés de sections selon la langue
    section_keywords = {
        'fr': ['article', 'section', 'clause', 'chapitre', 'annexe', 'préambule', 'définitions'],
        'en': ['article', 'section', 'clause', 'chapter', 'annex', 'preamble', 'definitions', 'whereas'],
        'ar': ['مادة', 'بند', 'فصل', 'ملحق', 'تعريفات', 'ديباجة', 'شرط']
    }
    
    keywords = section_keywords.get(language, section_keywords['en'])
    sections = {}
    current_section = "Introduction"
    section_content = []
    
    lines = content.split('\n')[:150]
    
    for line in lines:
        line = line.strip()
        if any(keyword.lower() in line.lower() for keyword in keywords):
            if section_content:
                sections[current_section] = '\n'.join(section_content)
            current_section = line[:150]
            section_content = []
        else:
            section_content.append(line)
    
    if section_content:
        sections[current_section] = '\n'.join(section_content)
    
    return {
        'sections': sections,
        'total_sections': len(sections),
        'language': language,
        'structure_quality': min(100, len(sections) * 15)
    }

@tool
def advanced_legal_domain_identifier(content: str, language: str = 'en') -> Dict[str, Any]:
    """Identificateur de domaine juridique multilingue avancé"""
    content = sanitize_multilingual_text(content.lower(), language)[:4000]
    
    # Domaines juridiques multilingues
    legal_domains = {
        'fr': {
            'commercial': ['vente', 'achat', 'commercial', 'livraison', 'paiement', 'facture', 'commande'],
            'emploi': ['employe', 'employeur', 'salaire', 'travail', 'contrat travail', 'congé', 'démission'],
            'confidentialite': ['confidentiel', 'secret', 'non-divulgation', 'propriétaire', 'nda'],
            'location': ['louer', 'bail', 'locataire', 'proprietaire', 'loyer', 'caution'],
            'service': ['service', 'prestation', 'consultant', 'freelance', 'mission'],
            'partenariat': ['partenaire', 'joint-venture', 'collaboration', 'association'],
            'assurance': ['assurance', 'prime', 'sinistre', 'couverture', 'police'],
            'intellectual_property': ['propriété intellectuelle', 'brevet', 'marque', 'copyright', 'licence']
        },
        'en': {
            'commercial': ['sale', 'purchase', 'commercial', 'delivery', 'payment', 'invoice', 'order'],
            'employment': ['employee', 'employer', 'salary', 'work', 'employment', 'vacation', 'termination'],
            'confidentiality': ['confidential', 'secret', 'non-disclosure', 'proprietary', 'nda'],
            'lease': ['lease', 'rent', 'tenant', 'landlord', 'rental', 'deposit'],
            'service': ['service', 'consulting', 'contractor', 'freelance', 'outsourcing'],
            'partnership': ['partner', 'joint-venture', 'collaboration', 'alliance'],
            'insurance': ['insurance', 'premium', 'claim', 'coverage', 'policy'],
            'intellectual_property': ['intellectual property', 'patent', 'trademark', 'copyright', 'license']
        },
        'ar': {
            'commercial': ['بيع', 'شراء', 'تجاري', 'تسليم', 'دفع', 'فاتورة', 'طلب'],
            'employment': ['موظف', 'صاحب عمل', 'راتب', 'عمل', 'عقد عمل', 'إجازة', 'إنهاء'],
            'confidentiality': ['سري', 'عدم إفشاء', 'ملكية', 'محظور'],
            'lease': ['إيجار', 'مستأجر', 'مالك', 'عقد إيجار', 'كراء'],
            'service': ['خدمة', 'استشارة', 'مقاول', 'مستقل'],
            'partnership': ['شراكة', 'مشروع مشترك', 'تعاون', 'تحالف'],
            'insurance': ['تأمين', 'قسط', 'مطالبة', 'تغطية', 'بوليصة'],
            'intellectual_property': ['ملكية فكرية', 'براءة', 'علامة تجارية', 'حقوق', 'ترخيص']
        }
    }
    
    domains = legal_domains.get(language, legal_domains['en'])
    domain_scores = {}
    
    for domain, keywords in domains.items():
        score = sum(1 for keyword in keywords if keyword in content)
        domain_scores[domain] = score
    
    best_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    total_score = sum(domain_scores.values())
    confidence = domain_scores[best_domain] / total_score if total_score > 0 else 0.5
    
    return {
        'domain': best_domain,
        'confidence': min(confidence, 1.0),
        'all_scores': domain_scores,
        'language': language,
        'complexity': min(100, total_score * 5)
    }

@tool
def advanced_risk_calculator(content: str, language: str = 'en') -> Dict[str, Any]:
    """Calculateur de risques avancé multilingue"""
    content = sanitize_multilingual_text(content.lower(), language)[:2000]
    
    # Indicateurs de risque multilingues
    risk_indicators = {
        'fr': {
            'high_risk': ['exclusion', 'limitation', 'responsabilité', 'pénalité', 'résiliation', 'force majeure'],
            'financial_risk': ['paiement', 'intérêts', 'amende', 'dommages', 'remboursement'],
            'legal_risk': ['tribunal', 'arbitrage', 'litige', 'juridiction', 'loi applicable'],
            'operational_risk': ['délai', 'performance', 'qualité', 'spécification', 'livraison']
        },
        'en': {
            'high_risk': ['exclusion', 'limitation', 'liability', 'penalty', 'termination', 'force majeure'],
            'financial_risk': ['payment', 'interest', 'fine', 'damages', 'refund', 'compensation'],
            'legal_risk': ['court', 'arbitration', 'dispute', 'jurisdiction', 'governing law'],
            'operational_risk': ['deadline', 'performance', 'quality', 'specification', 'delivery']
        },
        'ar': {
            'high_risk': ['استبعاد', 'حد', 'مسؤولية', 'غرامة', 'إنهاء', 'قوة قاهرة'],
            'financial_risk': ['دفع', 'فوائد', 'غرامة', 'أضرار', 'استرداد'],
            'legal_risk': ['محكمة', 'تحكيم', 'نزاع', 'اختصاص', 'قانون'],
            'operational_risk': ['موعد', 'أداء', 'جودة', 'مواصفات', 'تسليم']
        }
    }
    
    indicators = risk_indicators.get(language, risk_indicators['en'])
    risk_scores = {}
    
    for risk_type, keywords in indicators.items():
        score = sum(1 for keyword in keywords if keyword in content)
        risk_scores[risk_type] = min(100, score * 20)
    
    overall_risk = sum(risk_scores.values()) / len(risk_scores)
    
    return {
        'financial_risk': risk_scores.get('financial_risk', 30),
        'legal_risk': risk_scores.get('legal_risk', 25),
        'operational_risk': risk_scores.get('operational_risk', 20),
        'overall_risk': overall_risk,
        'risk_level': 'high' if overall_risk > 60 else 'medium' if overall_risk > 30 else 'low',
        'language': language
    }

@tool
def compliance_checker(content: str, language: str = 'en') -> Dict[str, Any]:
    """Vérificateur de conformité multilingue"""
    content = sanitize_multilingual_text(content.lower(), language)[:3000]
    
    # Standards de conformité par langue/région
    compliance_standards = {
        'fr': {
            'GDPR': ['données personnelles', 'consentement', 'droit oubli', 'responsable traitement'],
            'Code_Civil': ['bonne foi', 'équité', 'force obligatoire', 'résolution'],
            'Code_Commerce': ['commercial', 'société', 'registre commerce', 'tribunal commerce']
        },
        'en': {
            'GDPR': ['personal data', 'consent', 'right erasure', 'data controller', 'privacy'],
            'SOX': ['financial reporting', 'internal controls', 'auditing', 'compliance'],
            'Contract_Law': ['consideration', 'capacity', 'legality', 'mutual assent']
        },
        'ar': {
            'Islamic_Law': ['حلال', 'شرعي', 'ربا', 'غرر', 'عدالة'],
            'Civil_Code': ['حسن نية', 'عدالة', 'التزام', 'تعاقد'],
            'Commercial_Law': ['تجاري', 'شركة', 'سجل تجاري', 'محكمة تجارية']
        }
    }
    
    standards = compliance_standards.get(language, compliance_standards['en'])
    compliance_scores = {}
    gaps = []
    
    for standard, keywords in standards.items():
        score = sum(1 for keyword in keywords if keyword in content)
        compliance_scores[standard] = min(100, score * 25)
        
        if score == 0:
            gaps.append(standard)
    
    overall_compliance = sum(compliance_scores.values()) / len(compliance_scores)
    
    return {
        'overall_compliance': overall_compliance,
        'individual_scores': compliance_scores,
        'compliance_gaps': gaps,
        'compliance_level': 'high' if overall_compliance > 70 else 'medium' if overall_compliance > 40 else 'low',
        'language': language
    }

# Agent runner multilingue optimisé
@multilingual_retry_with_rate_limit(max_attempts=2, base_delay=0.3)
async def run_multilingual_agent_fast(agent, prompt, tools=None, language='en'):
    """Runner d'agent multilingue ultra-rapide"""
    try:
        # Optimiser la taille du prompt selon la langue
        max_prompt_size = 5000 if language == 'ar' else 4500
        if len(prompt) > max_prompt_size:
            prompt = prompt[:max_prompt_size] + f"... [content truncated for speed - {language}]"
        
        if tools:
            result = agent.run(prompt, tools=tools)
        else:
            result = agent.run(prompt)
        
        if hasattr(result, 'content') and result.content:
            return result.content
        elif hasattr(result, 'response') and result.response:
            return result.response
        else:
            return str(result)
    except Exception as e:
        logger.warning(f"Erreur agent multilingue, fallback: {str(e)}")
        return f"Partial analysis available. Error: {str(e)}"

def create_multilingual_agents(text_content: str, language: str = 'en'):
    """Création d'agents multilingues optimisés"""
    model = get_multilingual_cached_model()
    
    # Instructions adaptées selon la langue
    instructions_map = {
        'fr': {
            'structure': "Tu analyses rapidement la structure des contrats en français. Sois concis et précis.",
            'legal': "Tu identifies les risques juridiques en français. Concentre-toi sur les points critiques.",
            'negotiate': "Tu identifies les points négociables en français. Focus sur l'impact maximum.",
        },
        'en': {
            'structure': "You rapidly analyze contract structures in English. Be concise and precise.",
            'legal': "You identify legal risks in English. Focus on critical points.",
            'negotiate': "You identify negotiable points in English. Focus on maximum impact.",
        },
        'ar': {
            'structure': "أنت تحلل بنية العقود باللغة العربية بسرعة. كن موجزاً ودقيقاً.",
            'legal': "أنت تحدد المخاطر القانونية باللغة العربية. ركز على النقاط الحرجة.",
            'negotiate': "أنت تحدد النقاط القابلة للتفاوض باللغة العربية. ركز على التأثير الأقصى.",
        }
    }
    
    instructions = instructions_map.get(language, instructions_map['en'])
    
    # Agent structure multilingue
    structure_agent = Agent(
        model=model,
        name=f'Multilingual Structure Agent ({language})',
        role='Contract Structure Analyzer',
        instructions=dedent(f"""
            {instructions['structure']}
            
            Analyze contract structure and respond in {SUPPORTED_LANGUAGES.get(language, {}).get('name', language)}.
            
            Provide:
            1. Current structure (3-5 points max)
            2. Missing critical sections (2-3 max)
            3. Clarity score (0-100%)
            4. Language-specific recommendations
            
            Be brief and actionable.
        """),
        tools=[advanced_multilingual_section_analyzer],
        show_tool_calls=False,
        markdown=True,
    )
    
    # Agent juridique multilingue
    legal_agent = Agent(
        model=model,
        name=f'Multilingual Legal Agent ({language})',
        role='Legal Risk Analyzer',
        instructions=dedent(f"""
            {instructions['legal']}
            
            Identify legal risks and respond in {SUPPORTED_LANGUAGES.get(language, {}).get('name', language)}.
            
            For each critical issue:
            📄 **Clause**: "[text]"
            ⚖️ **Risk**: [brief description]
            🚨 **Level**: [High/Medium/Low]
            🏛️ **Category**: [Legal/Financial/Operational]
            
            Maximum 4-5 risks. Be direct and precise.
        """),
        tools=[advanced_legal_domain_identifier, advanced_risk_calculator, compliance_checker],
        show_tool_calls=False,
        markdown=True,
    )
    
    # Agent négociation multilingue
    negotiate_agent = Agent(
        model=model,
        name=f'Multilingual Negotiation Agent ({language})',
        role='Negotiation Strategist',
        instructions=dedent(f"""
            {instructions['negotiate']}
            
            Identify negotiable points and respond in {SUPPORTED_LANGUAGES.get(language, {}).get('name', language)}.
            
            For each point (max 4):
            📝 **Clause**: "[text]"
            💪 **Leverage Score**: [0-10]
            🔄 **Suggestion**: "[alternative]"
            🎯 **Priority**: [High/Medium/Low]
            💰 **Impact**: [Financial/Legal/Operational]
            
            Focus on maximum value creation.
        """),
        tools=[advanced_risk_calculator],
        show_tool_calls=False,
        markdown=True,
    )
    
    return structure_agent, legal_agent, negotiate_agent

def chunk_text_multilingual(text: str, language: str = 'en', max_tokens: int = 2500) -> List[str]:
    """Chunking optimisé pour différentes langues"""
    # Ajustement selon la langue
    if language == 'ar':
        max_tokens = int(max_tokens * 1.2)  # Caractères arabes prennent plus de tokens
    elif language == 'fr':
        max_tokens = int(max_tokens * 1.1)  # Français légèrement plus verbeux
    
    if len(text) <= max_tokens * 4:
        return [text]
    
    # Séparateurs selon la langue
    separators = {
        'ar': ['\n\n', '。', '؟', '！', '\n'],
        'fr': ['\n\n', '. ', '? ', '! ', '\n'],
        'en': ['\n\n', '. ', '? ', '! ', '\n']
    }
    
    seps = separators.get(language, separators['en'])
    
    # Division intelligente
    for separator in seps:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for part in parts:
                part_length = len(part)
                if current_length + part_length > max_tokens * 4:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = [part]
                        current_length = part_length
                    else:
                        chunks.append(part[:max_tokens * 4])
                else:
                    current_chunk.append(part)
                    current_length += part_length
            
            if current_chunk:
                chunks.append(separator.join(current_chunk))
            
            return chunks[:4]  # Maximum 4 chunks
    
    # Fallback: division brute
    chunk_size = max_tokens * 4
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)][:4]

class AdvancedMultilingualLegalAISystem:
    """Système Legal AI avancé et multilingue ultra-rapide"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.analysis_history = []
        
    async def analyze_contract_advanced_fast(self, file_path: str, analysis_depth: str = 'quick') -> Dict[str, Any]:
        """Analyse avancée multilingue ultra-rapide"""
        logger.info(f"🚀 Démarrage analyse avancée multilingue: {file_path}")
        
        max_chunks_map = {
            'quick': 1,
            'standard': 2,
            'comprehensive': 3,
            'expert': 4  # Nouveau mode expert
        }
        max_chunks = max_chunks_map.get(analysis_depth, 1)
        
        start_time = datetime.now()
        
        try:
            # Extraction multilingue du contenu
            _, text_content, detected_language = create_multilingual_knowledge_base(file_path)
            logger.info(f"Langue détectée: {detected_language}")
            
            # Agents multilingues optimisés
            structure_agent, legal_agent, negotiate_agent = create_multilingual_agents(text_content, detected_language)
            
            # Chunking multilingue optimisé
            text_chunks = chunk_text_multilingual(text_content, detected_language)[:max_chunks]
            logger.info(f"Traitement de {len(text_chunks)} sections en {detected_language}")
            
            # Analyses avancées en parallèle
            async def analyze_chunk_advanced_parallel(chunk_idx, chunk):
                logger.debug(f"Analyse avancée section {chunk_idx + 1} ({detected_language})")
                
                # Exécution en parallèle des analyses avancées
                tasks = [
                    run_multilingual_agent_fast(
                        structure_agent, 
                        f"Analyze structure:\n\n{chunk[:2500]}", 
                        [advanced_multilingual_section_analyzer],
                        detected_language
                    ),
                    run_multilingual_agent_fast(
                        legal_agent, 
                        f"Identify legal risks and compliance issues:\n\n{chunk[:2500]}", 
                        [advanced_legal_domain_identifier, advanced_risk_calculator, compliance_checker],
                        detected_language
                    ),
                    run_multilingual_agent_fast(
                        negotiate_agent, 
                        f"Find negotiation opportunities:\n\n{chunk[:2500]}", 
                        [advanced_risk_calculator],
                        detected_language
                    )
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                return {
                    'structure': results[0] if not isinstance(results[0], Exception) else f"Structure analysis unavailable ({detected_language})",
                    'legal': results[1] if not isinstance(results[1], Exception) else f"Legal analysis unavailable ({detected_language})", 
                    'negotiate': results[2] if not isinstance(results[2], Exception) else f"Negotiation analysis unavailable ({detected_language})",
                    'language': detected_language
                }
            
            # Traitement parallèle avancé de tous les chunks
            chunk_tasks = [analyze_chunk_advanced_parallel(i, chunk) for i, chunk in enumerate(text_chunks)]
            chunk_results = await asyncio.gather(*chunk_tasks)
            
            # Agrégation des résultats multilingues
            combined_structure = "\n\n".join([r['structure'] for r in chunk_results])
            combined_legal = "\n\n".join([r['legal'] for r in chunk_results])
            combined_negotiate = "\n\n".join([r['negotiate'] for r in chunk_results])
            
            # Analyse supplémentaire pour mode expert
            additional_analyses = {}
            if analysis_depth == 'expert':
                # Analyses expertes supplémentaires
                expert_tasks = [
                    self._perform_financial_analysis(text_content, detected_language),
                    self._perform_compliance_deep_dive(text_content, detected_language),
                    self._perform_comparative_analysis(text_content, detected_language)
                ]
                expert_results = await asyncio.gather(*expert_tasks, return_exceptions=True)
                
                additional_analyses = {
                    'financial_analysis': expert_results[0] if not isinstance(expert_results[0], Exception) else "Financial analysis unavailable",
                    'compliance_deep_dive': expert_results[1] if not isinstance(expert_results[1], Exception) else "Compliance analysis unavailable", 
                    'comparative_analysis': expert_results[2] if not isinstance(expert_results[2], Exception) else "Comparative analysis unavailable"
                }
            
            # Synthèse finale multilingue avancée
            language_name = SUPPORTED_LANGUAGES.get(detected_language, {}).get('name', detected_language)
            synthesis_prompt = f"""
            Advanced multilingual contract synthesis in {language_name}:
            
            DETECTED LANGUAGE: {detected_language} ({language_name})
            
            STRUCTURE ANALYSIS: {combined_structure[:1000]}
            LEGAL & COMPLIANCE: {combined_legal[:1000]}  
            NEGOTIATION STRATEGY: {combined_negotiate[:1000]}
            
            {"ADDITIONAL EXPERT ANALYSES: " + str(additional_analyses)[:800] if additional_analyses else ""}
            
            Create a comprehensive markdown report in {language_name} with:
            
            # Executive Summary (3-4 key points)
            ## Document Analysis
            - Language: {language_name}
            - Structure Quality Score
            - Legal Domain Classification
            
            ## Critical Risk Assessment (4-5 risks max)
            - Financial Risks
            - Legal & Compliance Risks  
            - Operational Risks
            
            ## Strategic Negotiation Points (4-5 points max)
            - High-Priority Items
            - Leverage Opportunities
            - Counter-Proposals
            
            ## Compliance & Regulatory Analysis
            - Regulatory Gaps
            - Compliance Recommendations
            
            {"## Expert Analysis Results" if additional_analyses else ""}
            {"- Financial Impact Assessment" if additional_analyses else ""}
            {"- Deep Compliance Review" if additional_analyses else ""}
            {"- Market Comparison" if additional_analyses else ""}
            
            ## Action Plan (5 prioritized actions)
            
            Respond entirely in {language_name}. Be comprehensive yet concise.
            """
            
            synthesis_agent = Agent(
                model=get_multilingual_cached_model(),
                name=f'Advanced Synthesis Agent ({detected_language})',
                role=f'Advanced Contract Synthesizer - {language_name}',
                instructions=f"Create comprehensive contract analysis reports in {language_name}. Be thorough, actionable, and culturally appropriate.",
                markdown=True
            )
            
            final_result = await run_multilingual_agent_fast(synthesis_agent, synthesis_prompt, language=detected_language)
            
        except Exception as e:
            logger.error(f"Erreur analyse avancée multilingue: {str(e)}")
            final_result = f"""
# Advanced Multilingual Contract Analysis Report

## Analysis Error
An error occurred during advanced analysis: {str(e)}

## Partial Information Available
- File processed: {Path(file_path).name}
- Content extracted: {len(text_content) if 'text_content' in locals() else 0} characters
- Detected language: {detected_language if 'detected_language' in locals() else 'unknown'}
- Analysis mode: {analysis_depth}
            """
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        analysis = {
            'timestamp': datetime.now(),
            'file': file_path,
            'result': final_result,
            'analysis_depth': analysis_depth,
            'analysis_time': analysis_time,
            'chunks_processed': len(text_chunks) if 'text_chunks' in locals() else 0,
            'detected_language': detected_language if 'detected_language' in locals() else 'unknown',
            'language_name': SUPPORTED_LANGUAGES.get(detected_language if 'detected_language' in locals() else 'en', {}).get('name', 'Unknown'),
            'model_used': MODEL_NAME,
            'advanced_features': True,
            'multilingual': True,
            'additional_analyses': list(additional_analyses.keys()) if additional_analyses else []
        }
        
        self.analysis_history.append(analysis)
        logger.info(f"✅ Analyse avancée multilingue terminée en {analysis_time:.1f}s")
        
        return analysis
    
    async def _perform_financial_analysis(self, text_content: str, language: str) -> str:
        """Analyse financière approfondie"""
        try:
            financial_agent = Agent(
                model=get_multilingual_cached_model(),
                name=f'Financial Analysis Agent ({language})',
                role='Financial Risk Analyst',
                instructions=f"""
                Perform detailed financial analysis in {SUPPORTED_LANGUAGES.get(language, {}).get('name', language)}.
                
                Focus on:
                - Payment terms and conditions
                - Financial penalties and incentives
                - Currency and exchange rate risks
                - Cost allocation and responsibility
                - Revenue impact assessment
                
                Provide specific recommendations for financial risk mitigation.
                """,
                markdown=True
            )
            
            prompt = f"Perform comprehensive financial analysis of this contract:\n\n{text_content[:3000]}"
            return await run_multilingual_agent_fast(financial_agent, prompt, language=language)
        except Exception as e:
            return f"Financial analysis error: {str(e)}"
    
    async def _perform_compliance_deep_dive(self, text_content: str, language: str) -> str:
        """Analyse de conformité approfondie"""
        try:
            compliance_agent = Agent(
                model=get_multilingual_cached_model(),
                name=f'Compliance Analysis Agent ({language})',
                role='Regulatory Compliance Expert',
                instructions=f"""
                Perform deep compliance analysis in {SUPPORTED_LANGUAGES.get(language, {}).get('name', language)}.
                
                Analyze compliance with:
                - Regional legal requirements
                - Industry-specific regulations
                - Data protection laws (GDPR, etc.)
                - Employment law compliance
                - Commercial law standards
                
                Identify gaps and provide specific remediation steps.
                """,
                tools=[compliance_checker],
                markdown=True
            )
            
            prompt = f"Perform comprehensive compliance analysis:\n\n{text_content[:3000]}"
            return await run_multilingual_agent_fast(compliance_agent, prompt, [compliance_checker], language)
        except Exception as e:
            return f"Compliance analysis error: {str(e)}"
    
    async def _perform_comparative_analysis(self, text_content: str, language: str) -> str:
        """Analyse comparative avec les standards du marché"""
        try:
            comparative_agent = Agent(
                model=get_multilingual_cached_model(),
                name=f'Comparative Analysis Agent ({language})',
                role='Market Standards Analyst',
                instructions=f"""
                Perform comparative market analysis in {SUPPORTED_LANGUAGES.get(language, {}).get('name', language)}.
                
                Compare contract terms against:
                - Industry standard practices
                - Market benchmarks
                - Best practices recommendations
                - Competitive positioning
                - Risk-reward balance
                
                Suggest improvements based on market standards.
                """,
                markdown=True
            )
            
            prompt = f"Compare this contract against market standards:\n\n{text_content[:3000]}"
            return await run_multilingual_agent_fast(comparative_agent, prompt, language=language)
        except Exception as e:
            return f"Comparative analysis error: {str(e)}"

    async def specialized_multilingual_analysis(self, file_path: str, focus_area: str, language: str = None) -> Dict[str, Any]:
        """Analyse spécialisée multilingue"""
        try:
            logger.info(f"🎯 Analyse spécialisée multilingue ({focus_area})")
            
            if language is None:
                _, text_content, detected_language = create_multilingual_knowledge_base(file_path)
            else:
                _, text_content, detected_language = create_multilingual_knowledge_base(file_path)
                detected_language = language  # Override detection
            
            specialized_prompts = {
                'ip': {
                    'en': "Focus on intellectual property clauses, ownership, licensing terms, and IP protection mechanisms",
                    'fr': "Concentrez-vous sur les clauses de propriété intellectuelle, propriété, termes de licence et mécanismes de protection PI",
                    'ar': "ركز على بنود الملكية الفكرية والملكية وشروط الترخيص وآليات حماية الملكية الفكرية"
                },
                'privacy': {
                    'en': "Analyze data protection provisions, privacy compliance, confidentiality clauses, and GDPR requirements",
                    'fr': "Analysez les dispositions de protection des données, conformité confidentialité, clauses de confidentialité et exigences RGPD",
                    'ar': "حلل أحكام حماية البيانات والامتثال للخصوصية وبنود السرية ومتطلبات اللائحة العامة"
                },
                'liability': {
                    'en': "Examine liability limitations, indemnification clauses, insurance requirements, and risk allocation",
                    'fr': "Examinez les limitations de responsabilité, clauses d'indemnisation, exigences d'assurance et allocation des risques",
                    'ar': "افحص قيود المسؤولية وبنود التعويض ومتطلبات التأمين وتوزيع المخاطر"
                },
                'payment': {
                    'en': "Review payment terms, penalties, financial obligations, currency provisions, and payment security",
                    'fr': "Examinez les conditions de paiement, pénalités, obligations financières, dispositions monétaires et sécurité de paiement",
                    'ar': "راجع شروط الدفع والغرامات والالتزامات المالية وأحكام العملة وأمان الدفع"
                },
                'termination': {
                    'en': "Analyze termination clauses, exit strategies, post-termination obligations, and contract dissolution",
                    'fr': "Analysez les clauses de résiliation, stratégies de sortie, obligations post-résiliation et dissolution du contrat",
                    'ar': "حلل بنود الإنهاء واستراتيجيات الخروج والالتزامات بعد الإنهاء وفسخ العقد"
                },
                'compliance': {
                    'en': "Deep dive into regulatory compliance, legal requirements, industry standards, and governance provisions",
                    'fr': "Plongée approfondie dans la conformité réglementaire, exigences légales, normes industrielles et dispositions de gouvernance",
                    'ar': "تعمق في الامتثال التنظيمي والمتطلبات القانونية ومعايير الصناعة وأحكام الحوكمة"
                }
            }
            
            prompt_text = specialized_prompts.get(focus_area, specialized_prompts['compliance']).get(detected_language, 
                specialized_prompts.get(focus_area, specialized_prompts['compliance'])['en'])
            
            specialized_agent = Agent(
                model=get_multilingual_cached_model(),
                name=f'Specialized Agent ({focus_area}) - {detected_language}',
                role=f'Specialized Contract Analyst - {focus_area.title()}',
                instructions=f"""
                Perform specialized analysis in {SUPPORTED_LANGUAGES.get(detected_language, {}).get('name', detected_language)}.
                
                {prompt_text}
                
                Provide:
                1. Specific findings related to {focus_area}
                2. Risk assessment for this domain
                3. Compliance requirements
                4. Improvement recommendations
                5. Best practices alignment
                
                Be thorough and actionable.
                """,
                tools=[advanced_legal_domain_identifier, advanced_risk_calculator, compliance_checker],
                markdown=True
            )
            
            analysis = await run_multilingual_agent_fast(
                specialized_agent, 
                f"{prompt_text}\n\nContract content:\n{text_content[:4000]}",
                [advanced_legal_domain_identifier, advanced_risk_calculator, compliance_checker],
                detected_language
            )
            
            return {
                'focus_area': focus_area,
                'analysis': analysis,
                'language': detected_language,
                'language_name': SUPPORTED_LANGUAGES.get(detected_language, {}).get('name', detected_language),
                'timestamp': datetime.now(),
                'specialized': True,
                'multilingual': True
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse spécialisée multilingue: {str(e)}")
            return {
                'focus_area': focus_area,
                'error': f"Specialized multilingual analysis error: {str(e)}",
                'language': detected_language if 'detected_language' in locals() else 'unknown',
                'timestamp': datetime.now()
            }

    def generate_advanced_multilingual_report(self, analysis: Dict[str, Any]) -> str:
        """Génération de rapport avancé multilingue"""
        language = analysis.get('detected_language', 'en')
        language_name = analysis.get('language_name', 'English')
        
        # Templates multilingues pour le rapport
        report_templates = {
            'fr': {
                'title': '# 📋 Rapport d\'Analyse Avancée Multilingue de Contrat',
                'subtitle': 'Propulsé par OpenAI GPT-4o avec IA Multilingue',
                'sections': {
                    'info': '## 📊 Informations d\'Analyse',
                    'performance': '## ⚡ Métriques de Performance',
                    'features': '## 🌍 Fonctionnalités Multilingues',
                    'analysis': '## 📑 Résultats d\'Analyse'
                }
            },
            'en': {
                'title': '# 📋 Advanced Multilingual Contract Analysis Report',
                'subtitle': 'Powered by OpenAI GPT-4o with Multilingual AI',
                'sections': {
                    'info': '## 📊 Analysis Information',
                    'performance': '## ⚡ Performance Metrics',
                    'features': '## 🌍 Multilingual Features',
                    'analysis': '## 📑 Analysis Results'
                }
            },
            'ar': {
                'title': '# 📋 تقرير تحليل العقد المتقدم متعدد اللغات',
                'subtitle': 'مدعوم بـ OpenAI GPT-4o-mini مع الذكاء الاصطناعي متعدد اللغات',
                'sections': {
                    'info': '## 📊 معلومات التحليل',
                    'performance': '## ⚡ مقاييس الأداء',
                    'features': '## 🌍 الميزات متعددة اللغات',
                    'analysis': '## 📑 نتائج التحليل'
                }
            }
        }
        
        template = report_templates.get(language, report_templates['en'])
        
        report = f"""
{template['title']}

**Date**: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}
**Fichier**: {Path(analysis['file']).name}
**Langue Détectée**: {language_name} ({language})
**Mode d'Analyse**: {analysis.get('analysis_depth', 'standard').upper()}
**Temps de Traitement**: {analysis.get('analysis_time', 0):.1f} secondes ⚡

{template['sections']['info']}
- **Sections Traitées**: {analysis.get('chunks_processed', 0)}
- **Modèle IA**: {analysis.get('model_used', MODEL_NAME)}
- **Fonctionnalités Avancées**: ✅ Activées
- **Support Multilingue**: ✅ {language_name}
- **Analyses Supplémentaires**: {', '.join(analysis.get('additional_analyses', [])) if analysis.get('additional_analyses') else 'Standard'}

{template['sections']['performance']}
- **Vitesse**: {analysis.get('analysis_time', 0):.1f}s (Ultra-rapide)
- **Précision Multilingue**: Optimisée pour {language_name}
- **Traitement Parallèle**: ✅ 4 agents simultanés
- **Détection Automatique**: ✅ Langue identifiée

{template['sections']['features']}
- **🌐 Langues Supportées**: Français, English, العربية
- **📝 Extraction Intelligente**: OCR multilingue avec Tesseract
- **🔍 Analyse Contextuelle**: Adaptée aux spécificités culturelles
- **⚖️ Conformité Régionale**: Standards locaux intégrés
- **💰 Analyse Financière**: Risques et opportunités
- **🏛️ Vérification Réglementaire**: Conformité automatique

{template['sections']['analysis']}

{analysis['result']}

---
*Généré par le Système IA Juridique Avancé Multilingue*  
*Analyse en {language_name} • Performance Optimisée • Conforme aux Standards Internationaux*
        """
        
        return report

def setup_advanced_multilingual_environment(api_key: str = None):
    """Configuration de l'environnement multilingue avancé"""
    if api_key:
        global OPENAI_API_KEY
        OPENAI_API_KEY = api_key
    
    logger.info("✅ Environnement IA Juridique Multilingue Avancé configuré")
    logger.info("🚀 Configuration avancée:")
    logger.info(f"  - Modèle: {MODEL_NAME} (Optimisé multilingue)")
    logger.info(f"  - Langues: {', '.join([info['name'] for info in SUPPORTED_LANGUAGES.values()])}")
    logger.info(f"  - Rate limit: {RATE_LIMIT_DELAY}s | Parallèle: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"  - OCR multilingue: Tesseract avec fra/eng/ara")
    logger.info(f"  - Détection automatique de langue activée")

async def main():
    """Fonction principale avancée multilingue"""
    try:
        setup_advanced_multilingual_environment()
        legal_system = AdvancedMultilingualLegalAISystem()
        
        contract_path = r"C:\Users\DELL\Pictures\mes papier(Documents)\ENSET\ENSA\testTechnique\rapport.pdf"
        contract_path = Path(contract_path)
        
        if not contract_path.exists():
            logger.error(f"Fichier non trouvé: {contract_path}")
            return
        
        print("🌍 Analyse Multilingue Avancée en cours...")
        print("⚡ Détection automatique de langue • Traitement ultra-rapide")
        
        # Test avec différents modes d'analyse
        for mode in ['quick', 'standard', 'expert']:
            print(f"\n🚀 Mode {mode.upper()}:")
            
            analysis = await legal_system.analyze_contract_advanced_fast(contract_path, mode)
            report = legal_system.generate_advanced_multilingual_report(analysis)
            
            print(f"📊 Langue détectée: {analysis.get('language_name', 'Unknown')}")
            print(f"⏱️ Temps: {analysis['analysis_time']:.1f}s")
            print(f"📄 Sections: {analysis['chunks_processed']}")
            
            # Sauvegarde avec nom multilingue
            lang_code = analysis.get('detected_language', 'en')
            filename = f'rapport_multilingue_{mode}_{lang_code}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"💾 Rapport sauvegardé: {filename}")
            
            if mode == 'quick':  # Afficher un exemple
                print(f"\n📖 Extrait du rapport ({analysis.get('language_name', 'Unknown')}):")
                print(report[:800] + "...")
        
        # Test analyse spécialisée multilingue
        print("\n🎯 Test analyses spécialisées multilingues:")
        for focus in ['ip', 'privacy', 'compliance', 'financial']:
            spec_analysis = await legal_system.specialized_multilingual_analysis(contract_path, focus)
            print(f"✅ Analyse {focus} ({spec_analysis.get('language_name', 'Unknown')}): {len(spec_analysis.get('analysis', ''))} chars")
        
        print(f"\n🎉 Analyse multilingue avancée terminée avec succès!")
        print(f"🌍 Support complet: Français • English • العربية")
        print(f"⚡ Performance: Ultra-rapide avec détection automatique")
        
    except Exception as e:
        logger.error(f"❌ Erreur analyse multilingue avancée: {str(e)}")
        print(f"\n❌ Erreur: {str(e)}")
        print("💡 Suggestions:")
        print("   - Vérifiez votre clé API OpenAI")
        print("   - Testez avec un fichier plus petit")
        print("   - Assurez-vous que Tesseract OCR est installé")

if __name__ == "__main__":
    asyncio.run(main())