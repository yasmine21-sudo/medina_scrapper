# ================================================
# openrouter_safe_request.py
# Module pour gérer les requêtes OpenRouter en toute sécurité
# ================================================
import time
import json
import requests
from requests.exceptions import SSLError, ConnectionError, Timeout, HTTPError

def safe_openrouter_request(API_URL, headers, payload, section_name, output_file, max_retries=5):
    """
    Envoie une requête POST à l'API OpenRouter avec gestion d'erreurs.
    Sauvegarde automatiquement les résultats intermédiaires dans un fichier texte.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)

            # Vérifie les erreurs HTTP
            response.raise_for_status()

            data = response.json()

            # ✅ Extraction du texte de réponse
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
            else:
                content = json.dumps(data, indent=2)

            # ✅ Sauvegarde automatique
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"===== {section_name.upper()} =====\n\n")
                f.write(content)
                f.write("\n\n✅ Section générée avec succès.\n")

            print(f"✅ Section '{section_name}' enregistrée dans {output_file}")
            return content

        except HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"⚠️ Trop de requêtes (429). Nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Erreur HTTP ({e.response.status_code}): {e}")
                break

        except (SSLError, ConnectionError, Timeout) as e:
            wait_time = 2 ** attempt
            print(f"⚠️ Erreur réseau ({type(e).__name__}) : {e}. Reconnexion dans {wait_time}s...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"❌ Erreur inattendue : {e}")
            break

    print(f"🚨 Impossible de générer la section '{section_name}' après {max_retries} essais.")
    return None
