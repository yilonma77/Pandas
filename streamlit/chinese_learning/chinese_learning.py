import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
from googletrans import Translator
import re

# Configuration de la page
st.set_page_config(
    page_title="🇨🇳 Chinese Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    h1 {
        margin-bottom: 0.5rem !important;
        font-size: 2.2rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        font-size: 1.5rem !important;
    }
    
    h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.2rem !important;
    }
    
    /* Chinese characters styling */
    .chinese-char {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .pinyin {
        font-size: 1.5rem;
        color: #667eea;
        text-align: center;
        margin: 5px 0;
    }
    
    .example-sentence {
        background: rgba(103, 126, 234, 0.1);
        padding: 10px;
        border-left: 3px solid #667eea;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .definition-box {
        background: rgba(103, 126, 234, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 4px;
    }
    
    /* Compact alerts */
    .stAlert {
        padding: 0.5rem 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(103,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialiser session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Fichiers de données
HISTORY_FILE = "chinese_history.json"
FAVORITES_FILE = "chinese_favorites.json"

# Fonctions utilitaires
def load_data(filename):
    """Charge les données depuis un fichier JSON"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_data(filename, data):
    """Sauvegarde les données dans un fichier JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def contains_chinese(text):
    """Vérifie si le texte contient des caractères chinois"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def get_character_breakdown(char):
    """Analyse basique d'un caractère chinois"""
    if not contains_chinese(char):
        return None
    
    # Informations basiques
    info = {
        'unicode': hex(ord(char)),
        'character': char,
        'stroke_count': 'N/A'  # Nécessiterait une base de données
    }
    return info

def translate_text(text, src='zh-CN', dest='fr'):
    """Traduit le texte du chinois vers le français"""
    try:
        translator = Translator()
        result = translator.translate(text, src=src, dest=dest)
        return result.text
    except Exception as e:
        return f"Erreur de traduction: {str(e)}"

def get_pinyin_approximation(text):
    """Approximation du pinyin (version simple)"""
    # Note: Pour un vrai pinyin, il faudrait une bibliothèque comme python-pinyin
    try:
        translator = Translator()
        result = translator.translate(text, src='zh-CN', dest='en')
        # Google Translate ne donne pas le pinyin directement
        return "Pinyin non disponible (installer python-pinyin pour la vraie prononciation)"
    except:
        return "N/A"

def add_to_history(word, translation, examples):
    """Ajoute une recherche à l'historique"""
    entry = {
        'word': word,
        'translation': translation,
        'examples': examples,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Ajouter au début de l'historique
    st.session_state.history.insert(0, entry)
    
    # Garder seulement les 50 dernières recherches
    st.session_state.history = st.session_state.history[:50]
    
    # Sauvegarder
    save_data(HISTORY_FILE, st.session_state.history)

def add_to_favorites(word, translation):
    """Ajoute un mot aux favoris"""
    entry = {
        'word': word,
        'translation': translation,
        'added_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Vérifier si déjà dans les favoris
    if not any(fav['word'] == word for fav in st.session_state.favorites):
        st.session_state.favorites.append(entry)
        save_data(FAVORITES_FILE, st.session_state.favorites)
        return True
    return False

# Charger les données au démarrage
st.session_state.history = load_data(HISTORY_FILE)
st.session_state.favorites = load_data(FAVORITES_FILE)

# Titre principal
st.title("🇨🇳 Chinese Learning Assistant")
st.caption("Pour apprenants niveau avancé (C1) • 高级中文学习助手")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Toggle thème
    theme_col1, theme_col2 = st.columns([1, 1])
    with theme_col1:
        if st.button("🌙 Dark" if st.session_state.theme == 'dark' else "☀️ Light", use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
    with theme_col2:
        st.caption(f"Theme: {st.session_state.theme.capitalize()}")
    
    st.markdown("---")
    
    # Mode de traduction
    translation_mode = st.selectbox(
        "🔄 Direction de traduction:",
        ["Chinois → Français", "Français → Chinois", "Chinois → Anglais"]
    )
    
    # Options d'affichage
    st.markdown("**📊 Affichage:**")
    show_examples = st.checkbox("Afficher des exemples", value=True)
    show_breakdown = st.checkbox("Décomposition caractères", value=True)
    
    st.markdown("---")
    
    # Statistiques
    st.markdown("**📈 Statistiques:**")
    st.metric("🔍 Recherches", len(st.session_state.history))
    st.metric("⭐ Favoris", len(st.session_state.favorites))
    
    st.markdown("---")
    
    # Raccourcis rapides
    st.markdown("**⚡ Raccourcis:**")
    if st.button("📜 Voir l'historique", use_container_width=True):
        st.session_state.show_history = True
    if st.button("⭐ Voir les favoris", use_container_width=True):
        st.session_state.show_favorites = True
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.history = []
        save_data(HISTORY_FILE, [])
        st.success("Historique effacé!")

# Zone principale
tabs = st.tabs(["🔍 Recherche", "📚 Exemples & Pratique", "📜 Historique", "⭐ Favoris"])

# Tab 1: Recherche principale
with tabs[0]:
    st.subheader("Recherche de mots et expressions")
    
    # Définir la source et destination selon le mode
    if translation_mode == "Chinois → Français":
        src_lang, dest_lang = 'zh-CN', 'fr'
        placeholder = "输入中文词汇... (ex: 学习, 努力, 成功)"
    elif translation_mode == "Français → Chinois":
        src_lang, dest_lang = 'fr', 'zh-CN'
        placeholder = "Entrez un mot français... (ex: apprendre, effort, succès)"
    else:
        src_lang, dest_lang = 'zh-CN', 'en'
        placeholder = "输入中文词汇... (ex: 学习, 努力, 成功)"
    
    # Input principal
    col1, col2 = st.columns([4, 1])
    with col1:
        search_word = st.text_input(
            "🔍 Mot ou expression:",
            placeholder=placeholder,
            key="search_input"
        )
    with col2:
        search_button = st.button("🔍 Rechercher", type="primary", use_container_width=True)
    
    # Exemples rapides
    st.caption("💡 Exemples rapides:")
    example_cols = st.columns(6)
    example_words = ["学习", "努力", "成功", "挑战", "机会", "经验"]
    for i, word in enumerate(example_words):
        with example_cols[i]:
            if st.button(word, key=f"example_{i}", use_container_width=True):
                search_word = word
                search_button = True
    
    # Traitement de la recherche
    if search_button and search_word:
        with st.spinner('🔄 Traduction en cours...'):
            # Traduction principale
            translation = translate_text(search_word, src=src_lang, dest=dest_lang)
            
            st.markdown("---")
            
            # Affichage du résultat principal
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Affichage du mot/caractère
                if contains_chinese(search_word):
                    st.markdown(f'<div class="chinese-char">{search_word}</div>', unsafe_allow_html=True)
                    
                    # Pinyin (approximation)
                    if len(search_word) <= 4:
                        st.markdown(f'<div class="pinyin">[Prononciation]</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chinese-char" style="font-size: 2.5rem;">{search_word}</div>', unsafe_allow_html=True)
            
            with col2:
                # Traduction
                st.markdown("### 📖 Traduction")
                st.markdown(f'<div class="definition-box"><h3>{translation}</h3></div>', unsafe_allow_html=True)
                
                # Bouton favoris
                if st.button("⭐ Ajouter aux favoris", key="add_fav"):
                    if add_to_favorites(search_word, translation):
                        st.success("✅ Ajouté aux favoris!")
                    else:
                        st.info("ℹ️ Déjà dans les favoris")
            
            # Décomposition des caractères (si option activée)
            if show_breakdown and contains_chinese(search_word):
                st.markdown("---")
                st.markdown("### 🔤 Décomposition des caractères")
                
                char_cols = st.columns(min(len(search_word), 6))
                for i, char in enumerate(search_word):
                    if i < 6:  # Limiter à 6 caractères
                        with char_cols[i]:
                            st.markdown(f"**{char}**")
                            info = get_character_breakdown(char)
                            if info:
                                st.caption(f"Unicode: {info['unicode']}")
            
            # Exemples de phrases (si option activée)
            if show_examples and contains_chinese(search_word):
                st.markdown("---")
                st.markdown("### 📝 Exemples d'utilisation")
                
                # Exemples prédéfinis (devraient venir d'une base de données)
                examples = [
                    {
                        'zh': f"我正在{search_word}新的技能。",
                        'fr': f"Je suis en train de {translation.lower()} de nouvelles compétences.",
                        'level': 'HSK 4'
                    },
                    {
                        'zh': f"这个{search_word}非常重要。",
                        'fr': f"Ce/Cette {translation.lower()} est très important(e).",
                        'level': 'HSK 3'
                    },
                    {
                        'zh': f"通过不断{search_word}，你会进步的。",
                        'fr': f"En {translation.lower()} constamment, tu progresseras.",
                        'level': 'HSK 5'
                    }
                ]
                
                for idx, example in enumerate(examples):
                    with st.expander(f"📌 Exemple {idx + 1} - {example['level']}", expanded=(idx == 0)):
                        st.markdown(f'<div class="example-sentence">', unsafe_allow_html=True)
                        st.markdown(f"**🇨🇳 中文:** {example['zh']}")
                        st.markdown(f"**🇫🇷 Français:** {example['fr']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Ajouter à l'historique
                add_to_history(search_word, translation, examples)

# Tab 2: Exemples et pratique
with tabs[1]:
    st.subheader("📚 Exemples de phrases et pratique")
    
    # Catégories de vocabulaire
    category = st.selectbox(
        "Choisir une catégorie:",
        ["💼 Business", "🎓 Académique", "🗣️ Conversation", "📰 Actualités", "🎭 Culture"]
    )
    
    # Exemples par catégorie
    if category == "💼 Business":
        business_vocab = [
            {"word": "谈判", "pinyin": "tán pàn", "translation": "négociation", "example": "我们需要进行商业谈判。"},
            {"word": "合作", "pinyin": "hé zuò", "translation": "coopération", "example": "双方达成了合作协议。"},
            {"word": "市场", "pinyin": "shì chǎng", "translation": "marché", "example": "这个产品有很大的市场潜力。"},
        ]
        
        for vocab in business_vocab:
            with st.expander(f"{vocab['word']} - {vocab['translation']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**字:** {vocab['word']}")
                    st.caption(f"Pinyin: {vocab['pinyin']}")
                with col2:
                    st.markdown(f"**Traduction:** {vocab['translation']}")
                st.markdown(f"**例句:** {vocab['example']}")
    
    elif category == "🎓 Académique":
        academic_vocab = [
            {"word": "研究", "pinyin": "yán jiū", "translation": "recherche", "example": "他正在做博士研究。"},
            {"word": "论文", "pinyin": "lùn wén", "translation": "thèse/article", "example": "我要写一篇学术论文。"},
            {"word": "分析", "pinyin": "fēn xī", "translation": "analyse", "example": "我们需要深入分析这个问题。"},
        ]
        
        for vocab in academic_vocab:
            with st.expander(f"{vocab['word']} - {vocab['translation']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**字:** {vocab['word']}")
                    st.caption(f"Pinyin: {vocab['pinyin']}")
                with col2:
                    st.markdown(f"**Traduction:** {vocab['translation']}")
                st.markdown(f"**例句:** {vocab['example']}")
    
    st.markdown("---")
    
    # Mini quiz
    st.subheader("🎯 Mini Quiz")
    st.caption("Testez vos connaissances!")
    
    quiz_word = st.selectbox(
        "Quelle est la traduction de '成功' ?",
        ["Réussite/Succès ✅", "Échec", "Effort", "Essai"]
    )
    
    if quiz_word == "Réussite/Succès ✅":
        st.success("🎉 Correct!")
    elif quiz_word != "Quelle est la traduction de '成功' ?":
        st.error("❌ Incorrect. La bonne réponse est: Réussite/Succès")

# Tab 3: Historique
with tabs[2]:
    st.subheader("📜 Historique des Recherches")
    
    if st.session_state.history:
        # Options de filtrage
        col1, col2 = st.columns([3, 1])
        with col1:
            search_filter = st.text_input("🔍 Filtrer l'historique:", placeholder="Rechercher...")
        with col2:
            if st.button("🗑️ Tout effacer", use_container_width=True):
                st.session_state.history = []
                save_data(HISTORY_FILE, [])
                st.rerun()
        
        st.markdown("---")
        
        # Afficher l'historique
        for idx, entry in enumerate(st.session_state.history):
            if not search_filter or search_filter.lower() in entry['word'].lower() or search_filter.lower() in entry['translation'].lower():
                with st.expander(f"{entry['word']} → {entry['translation']} • {entry['timestamp']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Mot:** {entry['word']}")
                        st.markdown(f"**Traduction:** {entry['translation']}")
                    with col2:
                        st.caption(f"📅 {entry['timestamp']}")
                        if st.button("⭐ Ajouter aux favoris", key=f"hist_fav_{idx}"):
                            add_to_favorites(entry['word'], entry['translation'])
                            st.success("Ajouté!")
    else:
        st.info("📭 Aucune recherche dans l'historique. Commencez par rechercher des mots!")

# Tab 4: Favoris
with tabs[3]:
    st.subheader("⭐ Mots Favoris")
    
    if st.session_state.favorites:
        # Options
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            sort_by = st.selectbox("Trier par:", ["Date d'ajout (récent)", "Date d'ajout (ancien)", "Alphabétique"])
        with col2:
            search_fav = st.text_input("🔍 Rechercher dans les favoris:", placeholder="Filtrer...")
        with col3:
            if st.button("📥 Export CSV", use_container_width=True):
                df = pd.DataFrame(st.session_state.favorites)
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 Télécharger",
                    data=csv,
                    file_name=f"chinese_favorites_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        
        # Trier les favoris
        sorted_favs = st.session_state.favorites.copy()
        if sort_by == "Date d'ajout (ancien)":
            sorted_favs.reverse()
        elif sort_by == "Alphabétique":
            sorted_favs = sorted(sorted_favs, key=lambda x: x['word'])
        
        # Afficher en grille
        cols = st.columns(3)
        for idx, fav in enumerate(sorted_favs):
            if not search_fav or search_fav.lower() in fav['word'].lower() or search_fav.lower() in fav['translation'].lower():
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"### {fav['word']}")
                        st.caption(f"📖 {fav['translation']}")
                        st.caption(f"📅 Ajouté: {fav['added_date']}")
                        if st.button("🗑️ Retirer", key=f"remove_fav_{idx}", use_container_width=True):
                            st.session_state.favorites.pop(idx)
                            save_data(FAVORITES_FILE, st.session_state.favorites)
                            st.rerun()
                    st.markdown("---")
    else:
        st.info("⭐ Aucun mot favori. Ajoutez des mots depuis la recherche!")

# Footer
st.markdown("---")
st.caption("🇨🇳 Chinese Learning Assistant • Niveau C1 • © 2026")
st.caption("💡 Astuce: Utilisez l'historique et les favoris pour réviser régulièrement!")
