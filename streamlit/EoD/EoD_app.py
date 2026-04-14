import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title='EoD Report', page_icon='📊', layout='wide')

DATA_FILE    = 'eod_notes.json'
HISTORY_FILE = 'notes_history.json'

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'taiwan': '', 'hong_kong': '', 'australia': '', 'japan': '', 'last_updated': None}

def save_data(data):
    data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append({'timestamp': data['last_updated'], 'data': data.copy()})
    history = history[-50:]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def build_table(n_cols, n_rows):
    """Génère un tableau Markdown vide avec n_cols colonnes et n_rows lignes de données."""
    col_name = lambda i: f"Col{i+1}"
    header    = "| " + " | ".join(col_name(i) for i in range(n_cols)) + " |"
    separator = "| " + " | ".join("------" for _ in range(n_cols)) + " |"
    row       = "| " + " | ".join("      " for _ in range(n_cols)) + " |"
    return "\n\n" + "\n".join([header, separator] + [row] * n_rows) + "\n"


def market_editor(label, flag, key, current_value):
    st.subheader(f'{flag} {label}')

    # Insertion de tableau personnalisé
    with st.expander('📋 Insérer un tableau', expanded=False):
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            n_cols = st.number_input('Colonnes', min_value=1, max_value=10, value=3,
                                     key=f'ncols_{key}', step=1)
        with c2:
            n_rows = st.number_input('Lignes', min_value=1, max_value=30, value=3,
                                     key=f'nrows_{key}', step=1)
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button('✅ Insérer', key=f'tbl_insert_{key}', use_container_width=True):
                current = st.session_state.get(key, current_value)
                st.session_state[key] = current + build_table(int(n_cols), int(n_rows))
                st.rerun()

    note = st.text_area(
        label,
        value=current_value,
        height=220,
        key=key,
        placeholder=(
            f'Notes EoD pour {label}...\n\n'
            'Markdown supporté :\n'
            '**gras**  *italique*  `code`\n\n'
            '| Ticker | Prix | Var |\n'
            '|--------|------|-----|\n'
            '| TSMC   |  920 | +1% |\n'
        ),
        label_visibility='collapsed'
    )
    if note.strip():
        with st.expander('👁️ Aperçu', expanded=False):
            st.markdown(note)
    return note

# En-tete
st.title('📊 EoD Report - Marchés Asiatiques')
st.markdown('---')
data = load_data()

# Sidebar
with st.sidebar:
    st.header('⚙️ Options')
    if data['last_updated']:
        st.info(f"📅 Dernière mise à jour:\n{data['last_updated']}")
    st.subheader('📊 Statistiques')
    for label, flag, k in [
        ('Taiwan',    '🇹🇼', 'taiwan'),
        ('Hong Kong', '🇭🇰', 'hong_kong'),
        ('Australia', '🇦🇺', 'australia'),
        ('Japan',     '🇯🇵', 'japan'),
    ]:
        st.metric(f'{flag} {label}', f"{len(data.get(k, ''))} car.")
    st.markdown('---')
    st.subheader('💾 Export')
    st.download_button(
        label='📥 JSON',
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime='application/json',
        use_container_width=True
    )
    df_export = pd.DataFrame([
        {'Marche': 'Taiwan',    'Notes': data.get('taiwan', '')},
        {'Marche': 'Hong Kong', 'Notes': data.get('hong_kong', '')},
        {'Marche': 'Australia', 'Notes': data.get('australia', '')},
        {'Marche': 'Japan',     'Notes': data.get('japan', '')},
    ])
    st.download_button(
        label='📊 CSV',
        data=df_export.to_csv(index=False, encoding='utf-8-sig'),
        file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True
    )
    st.markdown('---')
    st.caption('💡 **Markdown supporté**')
    st.code('**gras**  *italique*  `code`\n\n| Col | Val |\n|-----|-----|\n| A   |  1  |', language='markdown')

# 4 colonnes
col1, col2, col3, col4 = st.columns(4)
with col1:
    taiwan_note = market_editor('Taiwan',    '🇹🇼', 'taiwan_text', data.get('taiwan', ''))
with col2:
    hk_note     = market_editor('Hong Kong', '🇭🇰', 'hk_text',     data.get('hong_kong', ''))
with col3:
    aus_note    = market_editor('Australia', '🇦🇺', 'aus_text',    data.get('australia', ''))
with col4:
    japan_note  = market_editor('Japan',     '🇯🇵', 'japan_text',  data.get('japan', ''))

# Boutons
st.markdown('---')
col_btn1, col_btn2, col_btn3, _ = st.columns([2, 2, 2, 4])
with col_btn1:
    if st.button('💾 Sauvegarder', type='primary', use_container_width=True):
        save_data({'taiwan': taiwan_note, 'hong_kong': hk_note, 'australia': aus_note, 'japan': japan_note})
        st.success('✅ Données sauvegardées!')
        st.rerun()
with col_btn2:
    if st.button('🔄 Actualiser', use_container_width=True):
        st.rerun()
with col_btn3:
    if st.button('🗑️ Effacer tout', use_container_width=True):
        if st.session_state.get('confirm_delete', False):
            save_data({'taiwan': '', 'hong_kong': '', 'australia': '', 'japan': ''})
            st.session_state.confirm_delete = False
            st.success('🗑️ Notes effacées')
            st.rerun()
        else:
            st.session_state.confirm_delete = True
            st.warning('⚠️ Cliquez à nouveau pour confirmer')

# Historique
with st.expander('📜 Historique des modifications'):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        if history:
            st.write(f'**{len(history)} versions sauvegardées**')
            for i, version in enumerate(reversed(history[-10:])):
                st.text(f"Version {len(history) - i} : {version['timestamp']}")
                if st.button('Restaurer', key=f'restore_{i}'):
                    save_data(version['data'])
                    st.success('✅ Version restaurée!')
                    st.rerun()
                st.markdown('---')
        else:
            st.info('Aucun historique disponible')
    else:
        st.info('Aucun historique disponible')

st.markdown('---')
st.markdown('<div style="text-align:center"><p>📊 EoD Report · JSON local · © 2026</p></div>', unsafe_allow_html=True)
