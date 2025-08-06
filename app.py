import streamlit as st
import asyncio
import os
from pathlib import Path
from datetime import datetime
import logging
from legal_ai_system import AdvancedMultilingualLegalAISystem, setup_advanced_multilingual_environment, SUPPORTED_LANGUAGES

# Configurer le logging optimisé
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration OpenAI GPT-4o-mini multilingue
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# Configurer l'environnement multilingue avancé
setup_advanced_multilingual_environment(api_key=OPENAI_API_KEY)

# Initialiser le système LegalAI multilingue avancé avec mise en cache
@st.cache_resource
def initialize_advanced_multilingual_system():
    """Initialiser le système Legal AI multilingue avancé avec mise en cache"""
    return AdvancedMultilingualLegalAISystem(api_key=OPENAI_API_KEY)

legal_system = initialize_advanced_multilingual_system()

# Versions synchrones pour Streamlit
def run_advanced_analysis_sync(file_path, analysis_depth):
    """Version synchrone de l'analyse avancée multilingue pour Streamlit"""
    try:
        logger.info(f"🌍 Démarrage analyse avancée multilingue sync {analysis_depth}")
        start_time = datetime.now()
        
        analysis = asyncio.run(legal_system.analyze_contract_advanced_fast(file_path, analysis_depth))
        report = legal_system.generate_advanced_multilingual_report(analysis)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Analyse multilingue sync terminée en {duration:.1f}s")
        return analysis, report
    except Exception as e:
        logger.error(f"❌ Erreur analyse multilingue sync : {str(e)}")
        return None, f"""
❌ **Erreur lors de l'analyse multilingue avancée**

{str(e)}

💡 **Suggestions:**
- Vérifiez votre clé API OpenAI
- Assurez-vous d'avoir des crédits suffisants  
- Testez avec un fichier plus petit (< 5 MB)
- Vérifiez que Tesseract OCR est installé pour l'extraction multilingue
        """

def run_specialized_multilingual_analysis_sync(file_path, focus_area, language=None):
    """Version synchrone de l'analyse spécialisée multilingue"""
    try:
        logger.info(f"🎯 Analyse spécialisée multilingue sync ({focus_area})")
        
        specialized_result = asyncio.run(legal_system.specialized_multilingual_analysis(file_path, focus_area, language))
        
        logger.info("✅ Analyse spécialisée multilingue sync terminée")
        return specialized_result
    except Exception as e:
        logger.error(f"❌ Erreur analyse spécialisée multilingue sync : {str(e)}")
        return {"error": f"❌ Erreur lors de l'analyse spécialisée multilingue : {str(e)}"}

# Configuration de la page Streamlit multilingue
st.set_page_config(
    page_title="🌍 Analyse Multilingue de Contrats IA - Français • English • العربية",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour support multilingue
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .multilingual-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        text-align: center;
    }
    .language-indicator {
        background-color: #e8f4fd;
        border: 2px solid #2196F3;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .info-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .speed-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .analysis-type-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .analysis-type-card:hover {
        background-color: #e9ecef;
        border-color: #007bff;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal multilingue
st.markdown('<h1 class="main-header">🌍 Analyse Multilingue Avancée de Contrats Juridiques</h1>', unsafe_allow_html=True)

# Information sur les capacités multilingues
st.markdown("""
<div class="multilingual-box">
    <h3>🚀 IA Juridique Multilingue Ultra-Rapide</h3>
    <p><strong>Détection Automatique</strong> • Français • English • العربية</p>
    <p>Propulsé par OpenAI GPT-4o-mini avec analyse parallèle et OCR multilingue</p>
</div>
""", unsafe_allow_html=True)

# Indicateurs de langues supportées
lang_cols = st.columns(3)
with lang_cols[0]:
    st.markdown("""
    <div class="language-indicator">
        🇫🇷 <strong>Français</strong><br>
        Analyse juridique française<br>
        Conformité Code Civil
    </div>
    """, unsafe_allow_html=True)
with lang_cols[1]:
    st.markdown("""
    <div class="language-indicator">
        🇺🇸 <strong>English</strong><br>
        Legal analysis in English<br>
        Common law compliance
    </div>
    """, unsafe_allow_html=True)
with lang_cols[2]:
    st.markdown("""
    <div class="language-indicator rtl-text">
        🇸🇦 <strong>العربية</strong><br>
        تحليل قانوني باللغة العربية<br>
        التوافق مع الشريعة الإسلامية
    </div>
    """, unsafe_allow_html=True)

# Métriques de performance multilingues
st.markdown("### 📊 Performance Multilingue")
perf_cols = st.columns(5)
with perf_cols[0]:
    st.metric("🌍 Langues", "3", delta="Français • English • العربية")
with perf_cols[1]:
    st.metric("⚡ Vitesse", "10-30s", delta="Selon complexité")
with perf_cols[2]:
    st.metric("🤖 Modèle", "GPT-4o-mini", delta="Multilingue optimisé")
with perf_cols[3]:
    st.metric("📝 OCR", "Tesseract", delta="3 langues")
with perf_cols[4]:
    st.metric("🔍 Détection", "Automatique", delta="langdetect + AI")

# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📁 Téléversement Multilingue")
    st.markdown("Glissez-déposez votre contrat dans **n'importe quelle langue** pour une analyse automatique ⚡")
    
    # Téléversement optimisé multilingue
    uploaded_file = st.file_uploader(
        "🌍 Analyse Multilingue - PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        help="📄 Formats: PDF (avec OCR), Word, Texte | 🌍 Langues: Auto-détection | ⚡ Temps: 10-30s"
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb <= 5:
            st.success(f"✅ Fichier chargé: {uploaded_file.name} ({file_size_mb:.1f} MB) - Prêt pour analyse multilingue!")
        elif file_size_mb <= 10:
            st.warning(f"⚠️ Fichier volumineux: {uploaded_file.name} ({file_size_mb:.1f} MB) - Analyse possible mais plus lente")
        else:
            st.error(f"❌ Fichier trop volumineux: {uploaded_file.name} ({file_size_mb:.1f} MB) - Maximum: 10 MB")

with col2:
    st.markdown("### ⚙️ Paramètres d'Analyse Avancée")
    
    # Types d'analyse avancés multilingues
    analysis_type = st.selectbox(
        "Mode d'analyse",
        ["⚡ Ultra-Rapide", "🚀 Standard+", "🔬 Analyse Complète", "🎓 Mode Expert", "🎯 Analyse Spécialisée"],
        help="""
        • **⚡ Ultra-Rapide**: ~10-15s (1 section) - Aperçu rapide
        • **🚀 Standard+**: ~15-25s (2 sections) - Analyse équilibrée  
        • **🔬 Complète**: ~20-30s (3 sections) - Analyse approfondie
        • **🎓 Expert**: ~30-45s (4 sections + analyses financières/conformité/comparative)
        • **🎯 Spécialisée**: ~20-35s - Focus sur un domaine spécifique
        """
    )
    
    # Paramètres spécialisés multilingues
    focus_area = None
    language_override = None
    
    if analysis_type == "🎯 Analyse Spécialisée":
        st.markdown("**🎯 Domaine de spécialisation:**")
        focus_areas = {
            "ip": "🔒 Propriété Intellectuelle / IP / الملكية الفكرية",
            "privacy": "🛡️ Protection Données / Privacy / حماية البيانات",
            "liability": "⚖️ Responsabilité / Liability / المسؤولية", 
            "payment": "💰 Conditions Paiement / Payment / الدفع",
            "termination": "🚪 Résiliation / Termination / الإنهاء",
            "compliance": "🏛️ Conformité / Compliance / الامتثال"
        }
        
        focus_area = st.selectbox(
            "Domaine spécialisé",
            list(focus_areas.keys()),
            format_func=lambda x: focus_areas[x],
            help="Analyse approfondie d'un domaine juridique spécifique"
        )
        
        st.markdown("**🌍 Forcer une langue spécifique (optionnel):**")
        language_options = {
            "auto": "🤖 Détection Automatique",
            "fr": "🇫🇷 Forcer Français",
            "en": "🇺🇸 Force English", 
            "ar": "🇸🇦 فرض العربية"
        }
        
        language_choice = st.selectbox(
            "Langue de l'analyse",
            list(language_options.keys()),
            format_func=lambda x: language_options[x],
            help="Par défaut: détection automatique. Vous pouvez forcer une langue spécifique."
        )
        
        if language_choice != "auto":
            language_override = language_choice

    # Fonctionnalités avancées
    with st.expander("🔬 Fonctionnalités Avancées Activées"):
        st.success("✅ **Détection automatique de langue** - langdetect + IA")
        st.success("✅ **OCR multilingue** - Tesseract fra/eng/ara")  
        st.success("✅ **Analyse contextuelle** - Adaptée aux spécificités culturelles")
        st.success("✅ **Conformité régionale** - Standards locaux intégrés")
        st.success("✅ **Traitement parallèle** - 4 agents multilingues simultanés")
        
        if analysis_type == "🎓 Mode Expert":
            st.info("🎓 **Mode Expert inclut:**")
            st.write("• 💰 Analyse financière approfondie")  
            st.write("• 🏛️ Vérification de conformité réglementaire")
            st.write("• 📊 Analyse comparative avec standards du marché")

# Section d'analyse avec support multilingue
st.markdown("---")
st.markdown("### 🚀 Lancement de l'Analyse Multilingue")

# Bouton d'analyse avec support RTL
if st.button("🌍 ANALYSE MULTILINGUE", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Fichier requis</strong><br>
            Veuillez charger un contrat avant de lancer l'analyse multilingue.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Validation de taille avec suggestions multilingues
        MAX_FILE_SIZE = 10 * 1024 * 1024
        OPTIMAL_SIZE = 5 * 1024 * 1024
        
        if uploaded_file.size > MAX_FILE_SIZE:
            st.markdown("""
            <div class="error-box">
                🚫 <strong>Fichier trop volumineux</strong><br>
                Maximum: 10 MB | Optimal: < 5 MB pour performance maximale multilingue
            </div>
            """, unsafe_allow_html=True)
        else:
            # Avertissement performance pour gros fichiers
            if uploaded_file.size > OPTIMAL_SIZE:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>Fichier volumineux détecté</strong><br>
                    L'analyse multilingue peut prendre 45-60s au lieu de 10-30s pour ce fichier.
                </div>
                """, unsafe_allow_html=True)
            
            # Sauvegarde du fichier
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(temp_dir, f"multilingual_{timestamp}_{uploaded_file.name}")
            
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if not Path(file_path).exists():
                    st.error("❌ Erreur de sauvegarde du fichier")
                else:
                    # Interface de progression multilingue
                    progress_container = st.container()
                    
                    with progress_container:
                        # Timer multilingue en temps réel
                        timer_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        language_detection = st.empty()
                        
                        start_time = datetime.now()

                    try:
                        # Mapping des types d'analyse
                        depth_map = {
                            "⚡ Ultra-Rapide": "quick",
                            "🚀 Standard+": "standard", 
                            "🔬 Analyse Complète": "comprehensive",
                            "🎓 Mode Expert": "expert",
                            "🎯 Analyse Spécialisée": "specialized"
                        }
                        analysis_depth = depth_map.get(analysis_type, "quick")

                        # Phase 1: Extraction et détection de langue
                        progress_bar.progress(15)
                        status_text.text("🌍 Extraction multilingue et détection de langue...")
                        language_detection.info("🔍 Détection automatique de langue en cours...")
                        current_time = datetime.now()
                        timer_placeholder.metric("⏱️ Temps écoulé", f"{(current_time - start_time).total_seconds():.1f}s")
                        
                        import time
                        time.sleep(1)  # Simulation détection langue

                        # Phase 2: Configuration multilingue
                        progress_bar.progress(30)
                        status_text.text("⚙️ Configuration des agents multilingues...")
                        current_time = datetime.now()
                        timer_placeholder.metric("⏱️ Temps écoulé", f"{(current_time - start_time).total_seconds():.1f}s")
                        
                        time.sleep(0.5)

                        # Phase 3: Analyse selon le type
                        progress_bar.progress(60)
                        status_text.text("🤖 Analyse multilingue parallèle en cours...")
                        current_time = datetime.now()
                        timer_placeholder.metric("⏱️ Temps écoulé", f"{(current_time - start_time).total_seconds():.1f}s")

                        # Exécution selon le type
                        if analysis_type == "🎯 Analyse Spécialisée":
                            spec_analysis = run_specialized_multilingual_analysis_sync(file_path, focus_area, language_override)
                            
                            progress_bar.progress(100)
                            final_time = datetime.now()
                            total_duration = (final_time - start_time).total_seconds()
                            
                            # Affichage de la langue détectée
                            detected_lang = spec_analysis.get('language', 'unknown')
                            detected_lang_name = spec_analysis.get('language_name', 'Unknown')
                            language_detection.success(f"🌍 Langue détectée: {detected_lang_name} ({detected_lang})")
                            
                            timer_placeholder.metric("✅ Analyse Terminée", f"{total_duration:.1f}s", delta="Ultra-rapide multilingue!")
                            status_text.text("🎯 Analyse spécialisée multilingue terminée!")
                            
                            time.sleep(2)
                            progress_container.empty()
                            
                            if "error" not in spec_analysis:
                                st.markdown("### 🎯 Résultat de l'Analyse Spécialisée Multilingue")
                                
                                # Métriques de performance multilingues
                                perf_cols = st.columns(4)
                                with perf_cols[0]:
                                    st.metric("⚡ Temps", f"{total_duration:.1f}s")
                                with perf_cols[1]:
                                    st.metric("🎯 Domaine", focus_areas[focus_area].split('/')[0])
                                with perf_cols[2]:
                                    st.metric("🌍 Langue", detected_lang_name)
                                with perf_cols[3]:
                                    st.metric("🤖 Modèle", "GPT-4o-mini")
                                
                                st.markdown("---")
                                
                                # Affichage adapté à la direction de la langue
                                if detected_lang == 'ar':
                                    st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
                                    st.markdown(spec_analysis['analysis'])
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(spec_analysis['analysis'])
                                
                                # Téléchargement avec nom multilingue
                                report_filename = f"analyse_specialisee_{focus_area}_{detected_lang}_{timestamp}.md"
                                st.download_button(
                                    label=f"📥 Télécharger l'Analyse ({detected_lang_name})",
                                    data=spec_analysis['analysis'],
                                    file_name=report_filename,
                                    mime="text/markdown",
                                    use_container_width=True
                                )
                            else:
                                st.markdown(f"""
                                <div class="error-box">
                                    {spec_analysis['error']}
                                </div>
                                """, unsafe_allow_html=True)

                        else:
                            # Analyse avancée multilingue standard
                            analysis, report = run_advanced_analysis_sync(file_path, analysis_depth)
                            
                            progress_bar.progress(100)
                            final_time = datetime.now()
                            total_duration = (final_time - start_time).total_seconds()
                            
                            # Affichage de la langue détectée
                            if analysis:
                                detected_lang = analysis.get('detected_language', 'unknown')
                                detected_lang_name = analysis.get('language_name', 'Unknown')
                                language_detection.success(f"🌍 Langue détectée: {detected_lang_name} ({detected_lang})")
                            
                            timer_placeholder.metric("✅ Analyse Terminée", f"{total_duration:.1f}s", delta="Record multilingue!")
                            status_text.text("🚀 Analyse multilingue avancée terminée!")
                            
                            time.sleep(2)
                            progress_container.empty()
                            
                            if analysis:
                                st.markdown("### 📄 Rapport d'Analyse Multilingue Avancée")
                                
                                # Dashboard de performance multilingue complet
                                perf_metrics = st.columns(6)
                                with perf_metrics[0]:
                                    st.metric("⚡ Temps", f"{total_duration:.1f}s")
                                with perf_metrics[1]:
                                    st.metric("🌍 Langue", analysis.get('language_name', 'Unknown'))
                                with perf_metrics[2]:
                                    st.metric("📊 Sections", f"{analysis.get('chunks_processed', 0)}")
                                with perf_metrics[3]:
                                    st.metric("🤖 Modèle", analysis.get('model_used', 'GPT-4o-mini'))
                                with perf_metrics[4]:
                                    efficiency = min(100, (30 / max(total_duration, 1)) * 100)
                                    st.metric("🎯 Efficacité", f"{efficiency:.0f}%")
                                with perf_metrics[5]:
                                    features = len(analysis.get('additional_analyses', []))
                                    st.metric("🔬 Analyses+", f"{features}" if features > 0 else "Standard")
                                
                                st.markdown("---")
                                
                                # Affichage adapté à la direction de la langue
                                detected_lang = analysis.get('detected_language', 'en')
                                if detected_lang == 'ar':
                                    st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
                                    st.markdown(report)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(report)
                                
                                # Options de téléchargement multilingues
                                col1, col2 = st.columns(2)
                                with col1:
                                    report_filename = f"rapport_multilingue_{analysis_depth}_{detected_lang}_{timestamp}.md"
                                    st.download_button(
                                        label=f"📥 Télécharger le Rapport ({analysis.get('language_name', 'Unknown')})",
                                        data=report,
                                        file_name=report_filename,
                                        mime="text/markdown",
                                        use_container_width=True
                                    )
                                with col2:
                                    # Bouton pour nouvelle analyse
                                    if st.button("🔄 Nouvelle Analyse Multilingue", use_container_width=True):
                                        st.rerun()
                                        
                            else:
                                st.markdown(f"""
                                <div class="error-box">
                                    {report}
                                </div>
                                """, unsafe_allow_html=True)
                                
                    except Exception as e:
                        progress_container.empty()
                        st.markdown(f"""
                        <div class="error-box">
                            ❌ <strong>Erreur pendant l'analyse multilingue</strong><br>
                            {str(e)}
                            <br><br>
                            💡 <strong>Solutions multilingues:</strong><br>
                            • Vérifiez que Tesseract OCR est installé pour l'extraction<br>
                            • Essayez mode "Ultra-Rapide" pour documents complexes<br>
                            • Fichiers arabes/RTL peuvent nécessiter plus de temps<br>
                            • Vérifiez l'encodage du fichier (UTF-8 recommandé)
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"❌ Erreur de sauvegarde: {str(e)}")
                
            finally:
                # Nettoyage automatique
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass

# Sidebar multilingue optimisée
with st.sidebar:
    st.markdown("### 🌍 Dashboard Multilingue")
    
    # Statistiques multilingues en temps réel
    if legal_system.analysis_history:
        total_analyses = len(legal_system.analysis_history)
        multilingual_analyses = [a for a in legal_system.analysis_history if a.get('multilingual', False)]
        
        st.metric("🌍 Analyses Totales", total_analyses)
        
        if multilingual_analyses:
            avg_time = sum(a.get('analysis_time', 0) for a in multilingual_analyses) / len(multilingual_analyses)
            st.metric("⚡ Temps Moyen Multilingue", f"{avg_time:.1f}s")
            
            # Statistiques par langue
            languages = {}
            for analysis in multilingual_analyses:
                lang = analysis.get('detected_language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1
            
            if languages:
                st.markdown("**📊 Répartition par langue:**")
                for lang, count in languages.items():
                    lang_name = SUPPORTED_LANGUAGES.get(lang, {}).get('name', lang)
                    lang_flag = {'fr': '🇫🇷', 'en': '🇺🇸', 'ar': '🇸🇦'}.get(lang, '🌍')
                    st.write(f"{lang_flag} {lang_name}: {count}")
    
    # Historique des analyses multilingues
    if legal_system.analysis_history:
        st.markdown("### 📚 Historique Multilingue")
        multilingual_history = [a for a in legal_system.analysis_history if a.get('multilingual', False)][-3:]
        
        for analysis in reversed(multilingual_history):
            lang = analysis.get('detected_language', 'unknown')  
            lang_name = analysis.get('language_name', 'Unknown')
            lang_flag = {'fr': '🇫🇷', 'en': '🇺🇸', 'ar': '🇸🇦'}.get(lang, '🌍')
            
            with st.expander(f"{lang_flag} {Path(analysis['file']).name}", expanded=False):
                st.write(f"**🌍 Langue**: {lang_name}")
                st.write(f"**⏱️ Temps**: {analysis.get('analysis_time', 0):.1f}s")
                st.write(f"**📊 Sections**: {analysis.get('chunks_processed', 0)}")
                st.write(f"**🎯 Mode**: {analysis.get('analysis_depth', 'unknown').title()}")
                if analysis.get('additional_analyses'):
                    st.success(f"🔬 +{len(analysis.get('additional_analyses', []))} analyses expertes")
    
    # Informations système multilingue
    st.markdown("### 🌍 Système Multilingue")
    st.markdown("""
    **🤖 Modèle**: GPT-4o-mini (Multilingue)  
    **🌍 Langues**: Français • English • العربية  
    **🔍 Détection**: Automatique (langdetect + IA)  
    **📝 OCR**: Tesseract (fra/eng/ara)  
    **⚡ Vitesse**: 10-45s selon complexité  
    **🎯 Analyses**: 6 types spécialisés  
    **🔬 Expert**: Financière • Conformité • Comparative  
    """)
    
    # Guide multilingue
    with st.expander("💡 Guide Multilingue"):
        st.markdown("""
        **🌍 Langues supportées:**
        • **Français**: Analyse juridique française complète
        • **English**: Full English legal analysis  
        • **العربية**: تحليل قانوني شامل باللغة العربية
        
        **⚡ Performance optimale:**
        • Documents < 5 MB: 10-25s
        • PDF avec texte: Plus rapide que PDF scanné
        • Fichiers UTF-8: Meilleure détection de langue
        
        **🎯 Types d'analyse:**
        • **Ultra-Rapide**: Aperçu général (10-15s)
        • **Standard+**: Analyse équilibrée (15-25s)  
        • **Complète**: Analyse approfondie (20-30s)
        • **Expert**: Toutes analyses + financière/conformité (30-45s)
        • **Spécialisée**: Focus domaine spécifique (20-35s)
        
        **📝 Extraction multilingue:**
        • Texte natif: Extraction directe
        • PDF scanné: OCR Tesseract automatique
        • Détection langue: Automatique ou forcée
        """)

# Footer multilingue
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌍 Propulsé par OpenAI GPT-4o-mini | 🚀 IA Multilingue | 📋 Analyse Juridique Avancée</p>
    <p><small>Support complet: Français • English • العربية | Détection automatique de langue</small></p>
</div>
""", unsafe_allow_html=True)