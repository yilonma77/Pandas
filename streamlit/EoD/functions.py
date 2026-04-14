"""
EoD Report - Fonctions utilitaires
Formatage et envoi du rapport End of Day par email
"""

import json
import os
import smtplib
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ============================================================
# 🔧 CONFIGURATION — à remplir avec le même ID que EoD_app.py
# ============================================================
SPREADSHEET_ID   = "VOTRE_SPREADSHEET_ID_ICI"   # 👈 même ID que dans EoD_app.py
CREDENTIALS_FILE = "service_account.json"         # 👈 fichier JSON téléchargé depuis Google Cloud

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_gspread_client():
    """Crée un client gspread depuis le fichier service_account.json local."""
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(
            f"❌ Fichier '{CREDENTIALS_FILE}' introuvable.\n"
            "💡 Téléchargez-le depuis Google Cloud Console > IAM > Comptes de service"
        )
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return gspread.authorize(creds)


def load_data_from_sheets():
    """Charge les notes EoD depuis Google Sheets.
    
    Returns:
        dict: {taiwan, hong_kong, australia, japan, last_updated}
    """
    client = _get_gspread_client()
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    ws = spreadsheet.worksheet("Notes")
    records = ws.get_all_records()
    if records:
        row = records[0]
        return {
            "taiwan":       row.get("taiwan", ""),
            "hong_kong":    row.get("hong_kong", ""),
            "australia":    row.get("australia", ""),
            "japan":        row.get("japan", ""),
            "last_updated": row.get("last_updated", None),
        }
    return {"taiwan": "", "hong_kong": "", "australia": "", "japan": "", "last_updated": None}


def format_eod_notes():
    """Formate les notes EoD de manière professionnelle.
    
    Returns:
        tuple: (html_body, text_body) ou (None, message_erreur)
    """
    try:
        data = load_data_from_sheets()
    except Exception as e:
        return None, f"❌ Erreur lecture Google Sheets: {e}"

    if not any([data.get("taiwan"), data.get("hong_kong"), data.get("australia"), data.get("japan")]):
        return None, "❌ Aucune donnée disponible"

    # Date du jour
    today = datetime.now().strftime('%d %B %Y')
    timestamp = data.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Formater le rapport EoD - version HTML
    email_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .market-section {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .market-title {{
            color: #667eea;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        .market-content {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
        }}
        .empty-note {{
            color: #999;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #666;
            font-size: 12px;
        }}
        .stats {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .stats-item {{
            display: inline-block;
            margin: 5px 15px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 End of Day Report</h1>
        <h2>EoD Notes - Asian Markets</h2>
        <p>{today}</p>
    </div>

    <div class="stats">
        <strong>📈 Statistiques du rapport:</strong><br>
        <span class="stats-item">🇹🇼 Taiwan: {len(data.get('taiwan', ''))} caractères</span>
        <span class="stats-item">🇭🇰 Hong Kong: {len(data.get('hong_kong', ''))} caractères</span>
        <span class="stats-item">🇦🇺 Australia: {len(data.get('australia', ''))} caractères</span>
        <span class="stats-item">🇯🇵 Japan: {len(data.get('japan', ''))} caractères</span><br>
        <small>Dernière mise à jour: {timestamp}</small>
    </div>

    <div class="market-section">
        <div class="market-title">🇹🇼 TAIWAN</div>
        <div class="market-content">
            {data.get('taiwan', '<span class="empty-note">Aucune note saisie pour Taiwan</span>')}
        </div>
    </div>

    <div class="market-section">
        <div class="market-title">🇭🇰 HONG KONG</div>
        <div class="market-content">
            {data.get('hong_kong', '<span class="empty-note">Aucune note saisie pour Hong Kong</span>')}
        </div>
    </div>

    <div class="market-section">
        <div class="market-title">🇦🇺 AUSTRALIA</div>
        <div class="market-content">
            {data.get('australia', '<span class="empty-note">Aucune note saisie pour Australia</span>')}
        </div>
    </div>

    <div class="market-section">
        <div class="market-title">🇯🇵 JAPAN</div>
        <div class="market-content">
            {data.get('japan', '<span class="empty-note">Aucune note saisie pour Japan</span>')}
        </div>
    </div>

    <div class="footer">
        <p>📊 Rapport EoD généré automatiquement</p>
        <p>© 2026 - Toutes les données sont sauvegardées localement</p>
    </div>
</body>
</html>
"""

    # Version texte pour les clients email ne supportant pas HTML
    text_body = f"""
================================================================================
                    END OF DAY REPORT - {today}
                    EoD Notes - Asian Markets
================================================================================

Dernière mise à jour: {timestamp}

Statistiques:
  🇹🇼 Taiwan: {len(data.get('taiwan', ''))} caractères
  🇭🇰 Hong Kong: {len(data.get('hong_kong', ''))} caractères
  🇦🇺 Australia: {len(data.get('australia', ''))} caractères
  🇯🇵 Japan: {len(data.get('japan', ''))} caractères

================================================================================
🇹🇼 TAIWAN
================================================================================
{data.get('taiwan', 'Aucune note saisie pour Taiwan')}

================================================================================
🇭🇰 HONG KONG
================================================================================
{data.get('hong_kong', 'Aucune note saisie pour Hong Kong')}

================================================================================
🇦🇺 AUSTRALIA
================================================================================
{data.get('australia', 'Aucune note saisie pour Australia')}

================================================================================
🇯🇵 JAPAN
================================================================================
{data.get('japan', 'Aucune note saisie pour Japan')}

================================================================================
📊 Rapport EoD généré automatiquement | © 2026
================================================================================
"""

    return email_body, text_body


def send_eod_email(recipient_email, sender_email, sender_password):
    """
    Envoie le rapport EoD par email via Gmail.

    Parameters:
        recipient_email (str): Email du destinataire
        sender_email (str): Votre adresse Gmail
        sender_password (str): Mot de passe d'application Gmail (16 caractères)
                               ⚠️ PAS votre mot de passe habituel !
                               → myaccount.google.com > Sécurité > Mots de passe des applications

    Returns:
        bool: True si l'envoi a réussi, False sinon
    """

    html_body, text_body = format_eod_notes()

    if html_body is None:
        print(text_body)  # Affiche le message d'erreur
        return False

    # Configuration du message
    today = datetime.now().strftime('%Y-%m-%d')
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'📊 EoD Report - Asian Markets - {today}'
    msg['From'] = sender_email
    msg['To'] = recipient_email

    # Attacher les versions texte et HTML
    msg.attach(MIMEText(text_body, 'plain', 'utf-8'))
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))

    # Attacher les données en JSON (généré en mémoire depuis Google Sheets)
    try:
        data = load_data_from_sheets()
        json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        attachment = MIMEBase('application', 'json')
        attachment.set_payload(json_bytes)
        encoders.encode_base64(attachment)
        attachment.add_header(
            'Content-Disposition',
            f'attachment; filename=eod_notes_{today}.json'
        )
        msg.attach(attachment)
    except Exception:
        pass  # L'email est envoyé sans pièce jointe si Google Sheets est inaccessible

    try:
        print("🔄 Connexion au serveur SMTP...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        print("🔐 Authentification...")
        server.login(sender_email, sender_password)

        print("📧 Envoi de l'email...")
        server.send_message(msg)
        server.quit()

        print("✅ Email envoyé avec succès!")
        print(f"📬 Destinataire: {recipient_email}")
        print(f"📅 Date: {today}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("❌ ERREUR: Authentification échouée")
        print("💡 Utilisez un mot de passe d'application Gmail (pas votre mot de passe habituel)")
        print("💡 → myaccount.google.com > Sécurité > Mots de passe des applications")
        return False
    except Exception as e:
        print(f"❌ ERREUR lors de l'envoi: {str(e)}")
        return False
